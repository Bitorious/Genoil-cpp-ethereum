#include "ethash_cu_miner_kernel.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#define THREADS_PER_HASH (128 / 16)
#define HASHES_PER_LOOP (GROUP_SIZE / THREADS_PER_HASH)

#define FNV_PRIME	0x01000193

#define copy(dst, src, count) for (uint i = 0; i != count; ++i) { (dst)[i] = (src)[i]; }

#define countof(x) (sizeof(x) / sizeof(x[0]))

__device__ uint fnv(uint x, uint y)
{
	return x * FNV_PRIME ^ y;
}

__device__ uint4 fnv4(uint4 x, uint4 y)
{
	uint4 r; 
	r.x = x.x * FNV_PRIME ^ y.x;
	r.y = x.y * FNV_PRIME ^ y.y;
	r.z = x.z * FNV_PRIME ^ y.z;
	r.w = x.w * FNV_PRIME ^ y.w;
	return r;
}

__device__ uint fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}



__device__ hash64_t init_hash(hash32_t const* header, ulong nonce, uint isolate)
{
	hash64_t init;
	uint const init_size = countof(init.ulongs);
	uint const hash_size = countof(header->ulongs);

	// sha3_512(header .. nonce)
	ulong state[25];
	copy(state, header->ulongs, hash_size);
	state[hash_size] = nonce;
//	keccak_f1600_no_absorb(state, hash_size + 1, init_size, isolate);

	copy(init.ulongs, state, init_size);
	return init;
}

__device__ uint inner_loop(uint4 init, uint thread_id, uint* share, hash128_t const* g_dag, uint isolate)
{
	uint4 mix = init;

	// share init0
	if (thread_id == 0)
		*share = mix.x;
	__syncthreads();
	uint init0 = *share;

	uint a = 0;
	do
	{
		bool update_share = thread_id == (a / 4) % THREADS_PER_HASH;

#pragma unroll
		for (uint i = 0; i != 4; ++i)
		{
			if (update_share)
			{
				uint m[4] = { mix.x, mix.y, mix.z, mix.w };
				*share = fnv(init0 ^ (a + i), m[i]) % d_dag_size;
			}
			__syncthreads();

			mix = fnv4(mix, g_dag[*share].uint4s[thread_id]);
		}
	} while ((a += 4) != (d_acceses & isolate));

	return fnv_reduce(mix);
}

__device__ hash32_t final_hash(hash64_t const* init, hash32_t const* mix, uint isolate)
{
	ulong state[25];

	hash32_t hash;
	uint const hash_size = countof(hash.ulongs);
	uint const init_size = countof(init->ulongs);
	uint const mix_size = countof(mix->ulongs);

	// keccak_256(keccak_512(header..nonce) .. mix);
	copy(state, init->ulongs, init_size);
	copy(state + init_size, mix->ulongs, mix_size);
//	keccak_f1600_no_absorb(state, init_size + mix_size, hash_size, isolate);

	// copy out
	copy(hash.ulongs, state, hash_size);
	return hash;
}

typedef union
{
	struct
	{
		hash64_t init;
		uint pad; // avoid lds bank conflicts
	};
	hash32_t mix;
} compute_hash_share;

__device__ hash32_t compute_hash(
	compute_hash_share* share,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong nonce,
	uint isolate
	)
{
	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;

	// Compute one init hash per work item.
	hash64_t init = init_hash(g_header, nonce, isolate);

	// Threads work together in this phase in groups of 8.
	uint const thread_id = gid % THREADS_PER_HASH;
	uint const hash_id = (gid % d_workgroup_size) / THREADS_PER_HASH;

	hash32_t mix;
	uint i = 0;
	do
	{
		// share init with other threads
		if (i == thread_id)
			share[hash_id].init = init;
		__syncthreads();

		uint4 thread_init = share[hash_id].init.uint4s[thread_id % (64 / sizeof(uint4))];
		__syncthreads();

		uint thread_mix = inner_loop(thread_init, thread_id, share[hash_id].mix.uints, g_dag, isolate);

		share[hash_id].mix.uints[thread_id] = thread_mix;
		__syncthreads();

		if (i == thread_id)
			mix = share[hash_id].mix;
		__syncthreads();
	} while (++i != (THREADS_PER_HASH & isolate));

	return final_hash(&init, &mix, isolate);
}

//__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))

__global__ void ethash_search(
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
	)
{
	compute_hash_share * share = new compute_hash_share[d_workgroup_size / THREADS_PER_HASH];

	uint const gid = blockIdx.x * blockDim.x + threadIdx.x;
	hash32_t hash = compute_hash(share, g_header, g_dag, start_nonce + gid, isolate);

	

	if (__brevll(hash.ulongs[0]) < target)
	{
		uint slot = min(d_max_outputs, atomicInc(&g_output[0],1) + 1);
		g_output[slot] = gid;
	}
}

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
) 
{
}

void run_ethash_search(
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
)
{
	ethash_search<<<g_search_batch_size,g_workgroup_size>>>(g_output, g_header, g_dag, start_nonce,	target, isolate);
}

