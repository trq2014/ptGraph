//
// Created by gxl on 2021/3/24.
//

#include "bfsOpt.cuh"
#include "gpu_kernels.cuh"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
template<class T>
__global__ void testRefresh(bool* isActive, T* data, uint size) {
    streamVertices(size, [&](uint id) {
        data[id] = isActive[id];
    });
}

void bfs_opt(string path, uint sourceNode) {
    cout << "bfs_opt" << endl;
    GraphMeta<uint> graph;
    graph.setAlgType(BFS);
    graph.setSourceNode(25838548);
    graph.readDataFromFile(path, false);
    graph.setPrestoreRatio(0.95, 15);
    graph.initGraphHost();
    graph.initGraphDevice();
    cudaDeviceSynchronize();

    auto startTime = chrono::steady_clock::now();
    uint activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
    cout << "reduce " << duration << " micro s" << endl;
    cout << "activeNodesNum " << activeNodesNum << endl;
    gpuErrorcheck(cudaPeekAtLastError());
    vector<uint> testData(graph.vertexArrSize);

    thrust::device_ptr<bool> ptr_labeling(graph.isActiveD);
    startTime = chrono::steady_clock::now();
    uint sum = thrust::reduce(ptr_labeling, ptr_labeling + graph.vertexArrSize, 0, thrust::plus<uint>());
    //thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph.vertexArrSize, graph.prefixSumTemp, 0, thrust::plus<uint>());
    endTime = chrono::steady_clock::now();
    cudaMemcpy(testData.data(), graph.prefixSumTemp, graph.vertexArrSize * sizeof(uint), cudaMemcpyDeviceToHost);

    duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime).count();
    cout << "reduce " << duration << " micro s" << endl;
    int testTimes = 0;
    for (int testIndex = 0; testIndex < testTimes; testIndex++) {
        graph.refreshLabelAndValue();
        cudaDeviceSynchronize();
        activeNodesNum = reduceBool(graph.resultD, graph.isActiveD, graph.vertexArrSize, graph.grid, graph.block);
        int iter = 0;
        auto startProcessing = std::chrono::steady_clock::now();
        while (activeNodesNum) {
            iter++;
            setStaticAndOverloadLabelBool<<<graph.grid, graph.block>>>(graph.vertexArrSize, graph.isActiveD,
                                                                       graph.isStaticActive, graph.isOverloadActive,
                                                                       graph.isInStaticD);
            uint staticNodeNum = reduceBool(graph.resultD, graph.isInStaticD, graph.vertexArrSize, graph.grid, graph.block);
            if (staticNodeNum > 0) {
                //cout << "iter " << iter << " staticNodeNum " << staticNodeNum << endl;

            }
        }
    }
    gpuErrorcheck(cudaPeekAtLastError());
}