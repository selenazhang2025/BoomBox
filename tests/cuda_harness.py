"""Run the real CUDA kernel source on the CPU, so it can be tested without a GPU.

``kernels/boombust.cu`` is ordinary C++ once the CUDA keywords are defined away.
This harness prepends a shim that maps ``__global__``, ``__device__``,
``threadIdx``, ``__syncthreads`` and ``atomicAdd`` onto host equivalents --
std::barrier for the barrier and std::atomic_ref for the atomics, so the
multi-threaded run really is concurrent -- compiles the file with the system C++
compiler, and calls it through ctypes.

The kernel source itself is used verbatim: no copy, no reimplementation. What
this proves is the indexing, the shared-memory layout, the RNG and the tallying.
What it cannot prove is anything about warps, occupancy or real GPU memory.
"""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

KERNEL = Path(__file__).resolve().parents[1] / "kernels" / "boombust.cu"

SHIM = r"""
// ---- CUDA-to-host shim ---------------------------------------------------
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#define __device__
#define __forceinline__ inline
#define __global__
#define __restrict__
#define __shared__

struct bb_dim { int x; };
static bb_dim blockIdx{0};
static bb_dim blockDim{1};
static thread_local bb_dim threadIdx{0};

static std::barrier<>* bb_barrier_ptr = nullptr;
#define __syncthreads() do { if (bb_barrier_ptr) bb_barrier_ptr->arrive_and_wait(); } while (0)

static inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    return std::atomic_ref<unsigned long long>(*p).fetch_add(v);
}

// Dynamic shared memory. One block runs at a time, so a single buffer is enough.
// Declared with C linkage to match the kernel's own extern "C" block.
#define BB_SHARED_FLOATS 65536
extern "C" { float shared[BB_SHARED_FLOATS]; }
"""

DRIVER = r"""
// ---- host driver ---------------------------------------------------------
extern "C" int bb_shared_capacity() { return BB_SHARED_FLOATS; }

extern "C" void bb_launch(const float* my_params, const float* opp_params,
                          const int* lineups, const int* opp_lineup,
                          unsigned long long* wins, unsigned long long* ties,
                          int n_my, int n_opp, int n_lineups, int k, int k_opp,
                          long long n_sims, unsigned long long seed, int tile,
                          int blocks, int threads) {
    blockDim.x = threads;
    std::barrier<> bar(threads);
    bb_barrier_ptr = (threads > 1) ? &bar : nullptr;
    for (int b = 0; b < blocks; ++b) {
        blockIdx.x = b;
        if (threads == 1) {
            threadIdx.x = 0;
            simulate_lineups(my_params, opp_params, lineups, opp_lineup, wins, ties,
                             n_my, n_opp, n_lineups, k, k_opp, n_sims, seed, tile);
            continue;
        }
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([=]() {
                threadIdx.x = t;
                simulate_lineups(my_params, opp_params, lineups, opp_lineup, wins, ties,
                                 n_my, n_opp, n_lineups, k, k_opp, n_sims, seed, tile);
            });
        }
        for (auto& th : pool) th.join();
    }
    bb_barrier_ptr = nullptr;
}
"""

_CACHE: dict[str, ctypes.CDLL] = {}


def find_compiler() -> str | None:
    for name in ("c++", "clang++", "g++"):
        path = shutil.which(name)
        if path:
            return path
    return None


def build() -> ctypes.CDLL:
    """Compile the kernel source into a loadable shared library (cached)."""
    source = SHIM + KERNEL.read_text() + DRIVER
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    if digest in _CACHE:
        return _CACHE[digest]
    compiler = find_compiler()
    if compiler is None:
        raise RuntimeError("no C++ compiler on PATH")
    out_dir = Path(tempfile.gettempdir()) / "boombust_kernel_cpu"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / f"kernel_{digest}.cpp"
    lib = out_dir / f"kernel_{digest}.so"
    if not lib.exists():
        src.write_text(source)
        proc = subprocess.run(
            [compiler, "-std=c++20", "-O2", "-shared", "-fPIC", str(src), "-o", str(lib)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"compiling the kernel for the CPU failed:\n{proc.stderr[-4000:]}")
    dll = ctypes.CDLL(str(lib))
    dll.bb_launch.restype = None
    dll.bb_launch.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_longlong, ctypes.c_ulonglong, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
    ]
    dll.bb_shared_capacity.restype = ctypes.c_int
    _CACHE[digest] = dll
    return dll


def run_kernel(
    my_params: np.ndarray,
    opp_params: np.ndarray,
    lineups: np.ndarray,
    opp_lineup: np.ndarray,
    n_sims: int,
    seed: int = 0,
    tile: int = 16,
    threads: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Launch the kernel on the CPU and return (wins, ties)."""
    dll = build()
    my_params = np.ascontiguousarray(my_params, dtype=np.float32)
    opp_params = np.ascontiguousarray(opp_params, dtype=np.float32)
    lineups = np.ascontiguousarray(lineups, dtype=np.int32)
    opp_lineup = np.ascontiguousarray(opp_lineup, dtype=np.int32)

    n_my = my_params.shape[0]
    needed = n_my * (tile + 1) + tile
    capacity = dll.bb_shared_capacity()
    if needed > capacity:
        raise ValueError(f"shim shared buffer holds {capacity} floats, kernel needs {needed}")

    n_lineups, k = lineups.shape
    wins = np.zeros(n_lineups, dtype=np.uint64)
    ties = np.zeros(n_lineups, dtype=np.uint64)
    blocks = (int(n_sims) + tile - 1) // tile

    f32 = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.POINTER(ctypes.c_int)
    u64 = ctypes.POINTER(ctypes.c_ulonglong)
    dll.bb_launch(
        my_params.ctypes.data_as(f32), opp_params.ctypes.data_as(f32),
        lineups.ctypes.data_as(i32), opp_lineup.ctypes.data_as(i32),
        wins.ctypes.data_as(u64), ties.ctypes.data_as(u64),
        n_my, opp_params.shape[0], n_lineups, k, opp_lineup.size,
        int(n_sims), int(seed), int(tile), int(blocks), int(threads),
    )
    return wins.astype(np.int64), ties.astype(np.int64)
