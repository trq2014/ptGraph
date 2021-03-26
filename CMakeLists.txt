cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(ptGraph CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ptGraph main.cu gpu_kernels.cu gpu_kernels.cuh range.hpp bfs-sync.hpp tools.hpp common.cuh common.cu main.cuh bfs.cuh bfs.cu pagerank.cuh pagerank.cu cc.cuh cc.cu sssp.cuh sssp.cu GraphMeta.cu GraphMeta.cuh bfsOpt.cu bfsOpt.cuh constants.cuh)

set_target_properties(
        ptGraph
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(ptGraph -lcublas -lcurand)