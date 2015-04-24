#ifndef _ETHASH_CU_MINER_KERNEL_H_
#define _ETHASH_CU_MINER_KERNEL_H_

typedef unsigned long ulong;
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

void run_ethash_hash(
	__global__ hash32_t* g_hashes,
	__constant__ hash32_t const* g_header,
	__global__ hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
);

void run_ethash_search(
	__global__ volatile uint* restrict g_output,
	__constant__ hash32_t const* g_header,
	__global__ hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
);

#endif
