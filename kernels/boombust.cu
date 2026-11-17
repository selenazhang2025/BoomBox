// BoomBust lineup simulator.
//
// One block owns a tile of simulations. It generates every player's score for
// that tile into shared memory, reduces the opponent's starters to a single
// total per simulation, then has each thread sweep a stripe of candidate
// lineups and tally wins in registers. Only the win counters ever touch global
// memory, so the (L, S) totals array the NumPy and CuPy backends have to build
// in chunks is never materialised at all: device memory is O(L + P) instead of
// O(L * S).
//
// Randomness is counter-based (SplitMix64 over (seed, player, simulation)), so
// a thread can produce any draw with no state and no setup, and the NumPy
// reference in boombust/sim/rng.py can reproduce the same draws for testing.

extern "C" {

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ unsigned long long mix3(unsigned long long seed,
                                                   unsigned long long a,
                                                   unsigned long long b) {
    return splitmix64(splitmix64(seed ^ (a * 0x9E3779B97F4A7C15ULL)) ^ b);
}

// 24 random bits mapped into (0, 1). Never exactly zero, so logf() is safe.
__device__ __forceinline__ float u01(unsigned long long h) {
    // 0x1p-24f, not 1.0f/16777217.0f: 2^24+1 needs 25 mantissa bits, so it
    // rounds to 2^24 in float and the scale silently differs by 1 ULP from the
    // same expression evaluated in double. The NumPy reference uses this exact
    // constant, so every draw matches bit for bit. Top 24 bits + 1 keeps the
    // result in (0, 1] -- never 0, so logf() below is always safe.
    return (float)((h >> 40) + 1ULL) * 0x1p-24f;
}

// params: {p_active, ln_mu, ln_sigma}. Returns 0 when the player does not post
// a score, otherwise a log-normal draw.
__device__ __forceinline__ float player_draw(const float* __restrict__ params,
                                             unsigned long long seed,
                                             unsigned long long idx,
                                             unsigned long long sim) {
    const float p_active = params[0];
    const unsigned long long h = mix3(seed, idx, sim);
    if (u01(h) >= p_active) return 0.0f;
    const unsigned long long h2 = splitmix64(h ^ 0xD1B54A32D192ED03ULL);
    const float u1 = u01(h2);
    const float u2 = u01(splitmix64(h2));
    const float z = sqrtf(-2.0f * logf(u1)) * cosf((2.0f * u2) * 3.14159265358979323846f);
    return expf(params[1] + params[2] * z);
}

__global__ void simulate_lineups(const float* __restrict__ my_params,
                                 const float* __restrict__ opp_params,
                                 const int* __restrict__ lineups,
                                 const int* __restrict__ opp_lineup,
                                 unsigned long long* __restrict__ wins,
                                 unsigned long long* __restrict__ ties,
                                 const int n_my,
                                 const int n_opp,
                                 const int n_lineups,
                                 const int k,
                                 const int k_opp,
                                 const long long n_sims,
                                 const unsigned long long seed,
                                 const int tile) {
    extern __shared__ float shared[];
    // Stride is tile + 1 on purpose: with a stride that is a multiple of the
    // 32 shared-memory banks, every player's row would start in bank 0 and the
    // scattered reads in phase 3 would serialise 32 ways.
    const int stride = tile + 1;
    float* draws = shared;                                  // n_my x stride
    float* opp_total = shared + (size_t)n_my * stride;      // tile

    const long long base = (long long)blockIdx.x * (long long)tile;
    if (base >= n_sims) return;
    const long long remaining = n_sims - base;
    const int live = (remaining < (long long)tile) ? (int)remaining : tile;

    // Phase 1: my roster's scores for this tile of simulations.
    for (int e = threadIdx.x; e < n_my * live; e += blockDim.x) {
        const int p = e / live;
        const int t = e - p * live;
        draws[p * stride + t] =
            player_draw(&my_params[p * 3], seed, (unsigned long long)p, (unsigned long long)(base + t));
    }

    // Phase 2: the opponent's total. Their individual draws are never stored --
    // only the sum matters, so they live and die in registers.
    for (int t = threadIdx.x; t < live; t += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < k_opp; ++j) {
            const int q = opp_lineup[j];
            sum += player_draw(&opp_params[q * 3], seed,
                               (unsigned long long)q + 1000003ULL,
                               (unsigned long long)(base + t));
        }
        opp_total[t] = sum;
    }
    __syncthreads();

    // Phase 3: a stripe of candidate lineups per thread, tallied in registers.
    for (int l = threadIdx.x; l < n_lineups; l += blockDim.x) {
        const int* row = lineups + (size_t)l * k;
        unsigned long long won = 0ULL;
        unsigned long long tied = 0ULL;
        for (int t = 0; t < live; ++t) {
            float total = 0.0f;
            for (int j = 0; j < k; ++j) total += draws[row[j] * stride + t];
            const float other = opp_total[t];
            won += (total > other) ? 1ULL : 0ULL;
            tied += (total == other) ? 1ULL : 0ULL;
        }
        if (won) atomicAdd(&wins[l], won);
        if (tied) atomicAdd(&ties[l], tied);
    }
}

}  // extern "C"
