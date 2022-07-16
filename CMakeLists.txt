cmake_minimum_required(VERSION 3.10)
project(trtorch-test)

# Setting
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
list(APPEND CMAKE_PREFIX_PATH ../libtorch-1.11.0)
find_package(Torch REQUIRED)

# TRTorch
set(TRTORCH_INCLUDE ../torch_tensorrt/include)
file(GLOB TRTORCH_LIBRARIES ../torch_tensorrt/lib/*.so)

# get source
file(GLOB SRCS ./*.cpp)

add_executable(Test ${SRCS})
target_link_libraries(Test pthread ${TORCH_LIBRARIES} ${TRTORCH_LIBRARIES})
target_include_directories(Test PRIVATE ${TRTORCH_INCLUDE})
