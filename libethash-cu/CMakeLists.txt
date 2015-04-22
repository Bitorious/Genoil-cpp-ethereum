set(EXECUTABLE ethash-cu)

include(bin2h.cmake)
bin2h(SOURCE_FILE ethash_cu_miner_kernel.cu
	VARIABLE_NAME ethash_cu_miner_kernel
	HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/ethash_cu_miner_kernel.h)

aux_source_directory(. SRC_LIST)
file(GLOB HEADERS "*.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
add_library(${EXECUTABLE} ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${EXECUTABLE} ${CUDA_LIBRARIES} ethash)

install( TARGETS ${EXECUTABLE} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${EXECUTABLE} )

