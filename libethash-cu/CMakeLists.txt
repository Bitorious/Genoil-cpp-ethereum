set(EXECUTABLE ethash-cu)

<<<<<<< HEAD
aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52)

SET (CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)

CUDA_ADD_LIBRARY(${EXECUTABLE} STATIC ethash_cu_miner_kernel_globals.h ethash_cu_miner_kernel.h ethash_cu_miner_kernel.cu)

=======
include(bin2h.cmake)
bin2h(SOURCE_FILE ethash_cu_miner_kernel.cu
	VARIABLE_NAME ethash_cu_miner_kernel
	HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/ethash_cu_miner_kernel.h)

aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

>>>>>>> a019e9bba72d63f353fa7cd12dbd00441ce94c67
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${CUDA_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

