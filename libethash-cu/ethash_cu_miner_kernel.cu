#include "ethash_cu_miner_kernel.h"

void run_ethash_hash(
	__global__ hash32_t* g_hashes,
	__constant__ hash32_t const* g_header,
	__global__ hash128_t const* g_dag,
	ulong start_nonce,
	uint isolate
) 
{
}

void run_ethash_search(
	__global__ volatile uint* restrict g_output,
	__constant__ hash32_t const* g_header,
	__global__ hash128_t const* g_dag,
	ulong start_nonce,
	ulong target,
	uint isolate
)
{
