set(EXECUTABLE ethash-cl)

include(bin2h.cmake)
bin2h(SOURCE_FILE ethash_cl_miner_kernel.cl
	VARIABLE_NAME ethash_cl_miner_kernel
	HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/ethash_cl_miner_kernel.h)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52)


aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${OpenCL_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

