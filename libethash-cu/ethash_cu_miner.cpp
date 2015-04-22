/*
  This file is part of c-ethash.

  c-ethash is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  c-ethash is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/** @file ethash_cu_miner.cpp
* @author Tim Hughes <tim@twistedfury.com>
* @date 2015
*/


#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <assert.h>
#include <queue>
#include <vector>
#include <libethash/util.h>
#include <libethash/ethash.h>
#include "ethash_cu_miner.h"
#include "etash_cu_miner_kernel_globals.h"

#define ETHASH_BYTES 32

// workaround lame platforms
#if !CL_VERSION_1_2
#define CL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE
#define CL_MEM_HOST_READ_ONLY 0
#endif

#undef min
#undef max

using namespace std;

static void add_definition(std::string& source, char const* id, unsigned value)
{
	char buf[256];
	sprintf(buf, "#define %s %uu\n", id, value);
	source.insert(source.begin(), buf, buf + strlen(buf));
}

ethash_cu_miner::search_hook::~search_hook() {}

ethash_cu_miner::ethash_cu_miner()
{
}

std::string ethash_cu_miner::platform_info(unsigned _deviceId)
{
	int runtime_version;
	int device_count;
	
	device_count = get_num_devices();

	if (device_count == 0)
		return std::string();

	if (cudaRuntimeGetVersion(&runtime_version) == cudaErrorInvalidValue)
	{
		cout << cudaGetErrorString(cudaErrorInvalidValue) << endl;
		return std::string();
	}

	// use selected default device
	int device_num = std::min<int>((int)_deviceId, device_count - 1);

	cudaDeviceProp device_props;
	if (cudaGetDeviceProperties(&device_props, device_num) == cudaErrorInvalidDevice)
	{
		cout << cudaGetErrorString(cudaErrorInvalidDevice) << endl;
		return std::string();
	}

	char platform[5];
	int version_major = runtime_version / 1000;
	int version_minor = (runtime_version - (version_major * 1000)) / 10;
	sprintf(platform, "%d.%d", version_major, version_minor);

	char compute[5];
	sprintf(compute, "%d.%d", device_props.major, device_props.minor);

	return "{ \"platform\": \"CUDA " + std::string(platform) + "\", \"device\": \"" + device_props.name + "\", \"version\": \"Compute " + std::string(compute) + "\" }";
}

int ethash_cu_miner::get_num_devices()
{
	int device_count;

	if (cudaGetDeviceCount(&device_count) == cudaErrorNoDevice)
	{
		cout << cudaGetErrorString(cudaErrorNoDevice) << endl;
		return 0;
	}
	return device_count;
}

void ethash_cu_miner::finish()
{
}

bool ethash_cu_miner::init(ethash_params const& params, std::function<void(const char **, size_t& s)> _fillDAG, unsigned workgroup_size, unsigned _deviceId)
{
	// store params
	m_params = params;

	int device_count = get_num_devices();

	if (device_count == 0)
		return false;

	// use selected device
	int device_num = std::min<int>((int)_deviceId, device_count - 1);
	
	cudaDeviceProp device_props;
	if (cudaGetDeviceProperties(&device_props, device_num) == cudaErrorInvalidDevice)
	{
		cout << cudaGetErrorString(cudaErrorInvalidDevice) << endl;
		return false;
	}

	cout << "Using device: " << device_props.name << "(" << device_props.major << "." << device_props.minor << ")" << endl;

	cudaError_t r = cudaSetDevice(device_num);
	if (r != cudaSuccess)
	{
		cout << cudaGetErrorString(r) << endl;
		return false;
	}


	//m_context = cl::Context(std::vector<cl::Device>(&device, &device + 1));
	//m_queue = cl::CommandQueue(m_context, device);

	// use requested workgroup size, but we require multiple of 8
	m_workgroup_size = ((workgroup_size + 7) / 8) * 8;

	// patch source code

	g_dag_size = (unsigned)(params.full_size / ETHASH_MIX_BYTES);
	g_acceses = ETHASH_ACCESSES;
	g_max_outputs = c_max_search_results;

	g_workgroup_size = m_workgroup_size;
	g_search_batch_size  = c_search_batch_size;

	//debugf("%s", code.c_str());

	// create buffer for dag
	if (cudaMalloc(&m_dag_ptr, params.full_size) == cudaErrorMemoryAllocation) 
	{
		cout << cudaGetErrorString(cudaErrorMemoryAllocation) << endl;
		return false;
	}

	// create buffer for header
	if (cudaMalloc(&m_header, 32) == cudaErrorMemoryAllocation)
	{
		cout << cudaGetErrorString(cudaErrorMemoryAllocation) << endl;
		return false;
	}

	// copy dag to CPU.
	{
		// this is just ducttaped together. it feels overcomplicated for what it does. 
		const char * data;
		size_t size;
		_fillDAG(&data, size); // assumes dag is already in Host RAM. Needs some love.

		cudaError_t result = cudaMemcpy(m_dag_ptr, data, size, cudaMemcpyHostToDevice);
		if (result != cudaSuccess)
		{
			cout << cudaGetErrorString(result) << endl;
			return false;
		}
	}

	// create mining buffers
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		if (cudaMalloc(&m_hash_buf[i], 32 * c_hash_batch_size) == cudaErrorMemoryAllocation)
		{
			cout << cudaGetErrorString(cudaErrorMemoryAllocation) << endl;
			return false;
		}
		if (cudaMalloc(&m_search_buf[i], (c_max_search_results + 1) * sizeof(uint32_t)) == cudaErrorMemoryAllocation)
		{
			cout << cudaGetErrorString(cudaErrorMemoryAllocation) << endl;
			return false;
		}
	}
	return true;
}

void ethash_cu_miner::hash(uint8_t* ret, uint8_t const* header, uint64_t nonce, unsigned count)
{
	/*
	struct pending_batch
	{
		unsigned base;
		unsigned count;
		unsigned buf;
	};
	std::queue<pending_batch> pending;
	
	// update header constant buffer
	m_queue.enqueueWriteBuffer(m_header, true, 0, 32, header);

	/*
	__kernel void ethash_combined_hash(
		__global hash32_t* g_hashes,
		__constant hash32_t const* g_header,
		__global hash128_t const* g_dag,
		ulong start_nonce,
		uint isolate
		)
	*/
	/*
	m_hash_kernel.setArg(1, m_header);
	m_hash_kernel.setArg(2, m_dag);
	m_hash_kernel.setArg(3, nonce);
	m_hash_kernel.setArg(4, ~0u); // have to pass this to stop the compile unrolling the loop

	unsigned buf = 0;
	for (unsigned i = 0; i < count || !pending.empty(); )
	{
		// how many this batch
		if (i < count)
		{
			unsigned const this_count = std::min<unsigned>(count - i, c_hash_batch_size);
			unsigned const batch_count = std::max<unsigned>(this_count, m_workgroup_size);

			// supply output hash buffer to kernel
			m_hash_kernel.setArg(0, m_hash_buf[buf]);

			// execute it!
			m_queue.enqueueNDRangeKernel(
				m_hash_kernel,
				cl::NullRange,
				cl::NDRange(batch_count),
				cl::NDRange(m_workgroup_size)
				);
			m_queue.flush();
		
			pending.push({i, this_count, buf});
			i += this_count;
			buf = (buf + 1) % c_num_buffers;
		}

		// read results
		if (i == count || pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead, but this path isn't that important.
			uint8_t* hashes = (uint8_t*)m_queue.enqueueMapBuffer(m_hash_buf[batch.buf], true, CL_MAP_READ, 0, batch.count * ETHASH_BYTES);
			memcpy(ret + batch.base*ETHASH_BYTES, hashes, batch.count*ETHASH_BYTES);
			m_queue.enqueueUnmapMemObject(m_hash_buf[batch.buf], hashes);

			pending.pop();
		}
	}
	*/
}


void ethash_cu_miner::search(uint8_t const* header, uint64_t target, search_hook& hook)
{
	
	struct pending_batch
	{
		uint64_t start_nonce;
		unsigned buf;
	};
	std::queue<pending_batch> pending;

	static uint32_t const c_zero = 0;

	// update header constant buffer

	
	cudaMemcpy(m_header, header, 32, cudaMemcpyHostToDevice);
	for (unsigned i = 0; i != c_num_buffers; ++i)
	{
		cudaMemcpy(m_search_buf[i], &c_zero, 4, cudaMemcpyHostToDevice);
	}

	/*
	__kernel void ethash_combined_search(
		__global hash32_t* g_hashes,			// 0
		__constant hash32_t const* g_header,	// 1
		__global hash128_t const* g_dag,		// 2
		ulong start_nonce,						// 3
		ulong target,							// 4
		uint isolate							// 5
	)
	*/
	/*
	m_search_kernel.setArg(1, m_header);
	m_search_kernel.setArg(2, m_dag);

	// pass these to stop the compiler unrolling the loops
	m_search_kernel.setArg(4, target);
	m_search_kernel.setArg(5, ~0u);
	*/

	unsigned buf = 0;
	for (uint64_t start_nonce = 0; ; start_nonce += c_search_batch_size)
	{
		// supply output buffer to kernel
		//m_search_kernel.setArg(0, m_search_buf[buf]);
		//m_search_kernel.setArg(3, start_nonce);

		// execute it!
		
		run_ethash_search(m_search_buf[buf], m_header, m_dag_ptr, start_nonce, target, ~0u);

		pending.push({start_nonce, buf});
		buf = (buf + 1) % c_num_buffers;

		// read results
		if (pending.size() == c_num_buffers)
		{
			pending_batch const& batch = pending.front();

			// could use pinned host pointer instead
			uint32_t* results;
			cudaMemcpy(m_search_buf[batch.buf], results, (1 + c_max_search_results) * sizeof(uint32_t), cudaMemcpyDeviceToHost);

			unsigned num_found = std::min<unsigned>(results[0], c_max_search_results);

			uint64_t nonces[c_max_search_results];
			for (unsigned i = 0; i != num_found; ++i)
			{
				nonces[i] = batch.start_nonce + results[i+1];
			}
			
			bool exit = num_found && hook.found(nonces, num_found);
			exit |= hook.searched(batch.start_nonce, c_search_batch_size); // always report searched before exit
			if (exit)
				break;

			// reset search buffer if we're still going
			if (num_found)
				cudaMemcpy(m_search_buf[batch.buf], &c_zero, 4, cudaMemcpyHostToDevice);

			pending.pop();
		}
	}	
}

