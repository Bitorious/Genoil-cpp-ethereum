#ifndef _ETHASH_CU_MINER_KERNEL_H_
#define _ETHASH_CU_MINER_KERNEL_H_

typedef unsigned long long int ulong;
typedef unsigned int  uint;

typedef union
{
	ulong ulongs[32 / sizeof(ulong)];
	uint uints[32 / sizeof(uint)];
} hash32_t;

typedef union
{
	ulong ulongs[64 / sizeof(ulong)];
	uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

typedef union
{
	uint uints[128 / sizeof(uint)];
	uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

uint g_search_batch_size;
uint g_workgroup_size;

__device__ __constant__ uint d_workgroup_size;
__device__ __constant__ uint d_dag_size;
__device__ __constant__ uint d_acceses;
__device__ __constant__ uint d_max_outputs;

void run_ethash_hash(
	hash32_t* g_hashes,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
);

void run_ethash_search(
	uint* g_output,
	hash32_t const* g_header,
	hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
);

#endif
