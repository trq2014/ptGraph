//
// Created by gxl on 2021/2/1.
//

#ifndef PTGRAPH_GRAPHMETA_CUH
#define PTGRAPH_GRAPHMETA_CUH
#include <string>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
enum ALG_TYPE {
    BFS, SSSP, CC, PR
};
typedef uint SIZE_TYPE;
struct PartEdgeListInfo {
    SIZE_TYPE partActiveNodeNums;
    SIZE_TYPE partEdgeNums;
    SIZE_TYPE partStartIndex;
};

using namespace std;
template <class EdgeType>
class TestMeta {
public:
    ~TestMeta();
};

template<class EdgeType>
TestMeta<EdgeType>::~TestMeta() {

}

template <class EdgeType>
class GraphMeta {
public:
    dim3 grid = dim3(56, 1, 1);
    dim3 block = dim3(1024, 1, 1);
    SIZE_TYPE partOverloadSize;
    SIZE_TYPE overloadSize;
    SIZE_TYPE sourceNode = 0;
    SIZE_TYPE vertexArrSize;
    SIZE_TYPE edgeArrSize;
    SIZE_TYPE* nodePointers;
    EdgeType* edgeArray;
    //special for pr
    SIZE_TYPE* outDegree;
    SIZE_TYPE* degree;
    bool* label;
    float *valuePr;
    SIZE_TYPE * value;
    bool *isInStatic;
    SIZE_TYPE *overloadNodeList;
    SIZE_TYPE *staticNodePointer;
    SIZE_TYPE *activeNodeList;
    SIZE_TYPE *activeOverloadNodePointers;
    vector<PartEdgeListInfo> partEdgeListInfoArr;
    SIZE_TYPE *overloadEdgeList;
    //GPU
    uint *resultD;
    cudaStream_t steamStatic, streamDynamic;
    uint *prefixSumTemp;
    SIZE_TYPE *staticEdgeListD;
    SIZE_TYPE *overloadEdgeListD;
    bool *isInStaticD;
    SIZE_TYPE *overloadNodeListD;
    SIZE_TYPE *staticNodePointerD;
    SIZE_TYPE *degreeD;
    SIZE_TYPE *outDegreeD;
    // async need two labels
    bool *isActiveD;
    bool *isStaticActive;
    bool *isOverloadActive;
    SIZE_TYPE *valueD;
    float *valuePrD;
    float *sumD;
    SIZE_TYPE *activeNodeListD;
    SIZE_TYPE *activeNodeLabelingPrefixD;
    SIZE_TYPE *overloadLabelingPrefixD;
    SIZE_TYPE *activeOverloadNodePointersD;
    SIZE_TYPE *activeOverloadDegreeD;
    double adviseK;
    int paramSize;
    ALG_TYPE algType;
    void readDataFromFile(const string& fileName, bool isPagerank);
    ~GraphMeta();
    void setPrestoreRatio(double adviseK, int paramSize) {
        this->adviseK = adviseK;
        this->paramSize = paramSize;
    }
    void initGraphHost();
    void initGraphDevice();
    void refreshLabelAndValue();
    void initAndSetStaticNodePointers();
    void setAlgType(ALG_TYPE type) {
        algType = type;
    }
    void setSourceNode(SIZE_TYPE sourceNode) {
        this->sourceNode = sourceNode;
    }

private:
    SIZE_TYPE max_partition_size;
    SIZE_TYPE max_static_node;
    SIZE_TYPE total_gpu_size;
    uint fragmentSize = 4096;
    void getMaxPartitionSize();
    void initLableAndValue();
};
template<class EdgeType>
void GraphMeta<EdgeType>::readDataFromFile(const string &fileName, bool isPagerank) {
    cout << "====== readDataFromFile ============" << endl;
    auto startTime = chrono::steady_clock::now();
    ifstream infile(fileName, ios::in | ios::binary);
    infile.read((char *) &this->vertexArrSize, sizeof(uint));
    infile.read((char *) &this->edgeArrSize, sizeof(uint));
    cout << "vertex num: " << this->vertexArrSize << " edge num: " << this->edgeArrSize << endl;
    outDegree = new uint[vertexArrSize];
    if (isPagerank) {
        infile.read((char *) outDegree, sizeof(uint) * vertexArrSize);
    }
    nodePointers = new uint[vertexArrSize];
    infile.read((char *) nodePointers, sizeof(uint) * vertexArrSize);
    edgeArray = new EdgeType[edgeArrSize];
    infile.read((char *) edgeArray, sizeof(EdgeType) * edgeArrSize);
    infile.close();
    auto endTime = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
    cout << "readDataFromFile " << duration << " ms" << endl;
}

template<class EdgeType>
GraphMeta<EdgeType>::~GraphMeta() {
    cout << "~GraphMeta" << endl;
    delete[] nodePointers;
    delete[] outDegree;
    delete[] edgeArray;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initGraphHost() {
    cout << "========== initGraphHost ==========" << endl;
    degree = new SIZE_TYPE[vertexArrSize];
    isInStatic = new bool[vertexArrSize];
    overloadNodeList = new SIZE_TYPE[vertexArrSize];
    activeNodeList = new SIZE_TYPE[vertexArrSize];
    activeOverloadNodePointers = new SIZE_TYPE[vertexArrSize];

    for (SIZE_TYPE i = 0; i < vertexArrSize - 1; i++) {
        if (nodePointers[i] > edgeArrSize) {
            cout << i << "   " << nodePointers[i] << endl;
            break;
        }
        degree[i] = nodePointers[i + 1] - nodePointers[i];
    }
    degree[vertexArrSize - 1] = edgeArrSize - nodePointers[vertexArrSize - 1];
    getMaxPartitionSize();
    initLableAndValue();
    overloadEdgeList = (SIZE_TYPE *) malloc(overloadSize * sizeof(SIZE_TYPE));
}


template<class EdgeType>
void GraphMeta<EdgeType>::initGraphDevice() {
    cout << "=============== initGraphDevice ==============" << endl;
    cudaMalloc(&resultD, grid.x * sizeof(uint));
    cudaMalloc(&prefixSumTemp, vertexArrSize * sizeof(uint));
    uint* tempResult = new uint[grid.x];
    memset(tempResult, 0, sizeof(int) * grid.x);
    cudaMemcpy(resultD, tempResult, grid.x * sizeof(int), cudaMemcpyHostToDevice);

    gpuErrorcheck(cudaPeekAtLastError());
    //cudaMemset(resultD, 0, grid.x * sizeof(uint));

    cudaStreamCreate(&steamStatic);
    cudaStreamCreate(&streamDynamic);
    //pre store
    cudaMalloc(&staticEdgeListD, max_partition_size * sizeof(EdgeType));
    cudaMemcpy(staticEdgeListD, edgeArray, max_partition_size * sizeof(EdgeType), cudaMemcpyHostToDevice);

    cudaMalloc(&isInStaticD, vertexArrSize * sizeof(bool));
    cudaMalloc(&overloadNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers,vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMalloc(&overloadEdgeListD, partOverloadSize * sizeof(SIZE_TYPE));
    cudaMalloc(&degreeD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&isActiveD, vertexArrSize * sizeof(bool));
    cudaMalloc(&isStaticActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&isOverloadActive, vertexArrSize * sizeof(bool));
    cudaMalloc(&valueD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeNodeLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&overloadLabelingPrefixD, vertexArrSize * sizeof(SIZE_TYPE));

    cudaMalloc(&activeNodeListD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeOverloadNodePointersD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&activeOverloadDegreeD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(degreeD, degree, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool));
    cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool));

}

template<class EdgeType>
void GraphMeta<EdgeType>::initAndSetStaticNodePointers() {
    staticNodePointer = new uint[vertexArrSize];
    memcpy(staticNodePointer, nodePointers, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMalloc(&staticNodePointerD, vertexArrSize * sizeof(SIZE_TYPE));
    cudaMemcpy(staticNodePointerD, nodePointers, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);


}


template<class EdgeType>
void GraphMeta<EdgeType>::getMaxPartitionSize() {
    int deviceID;
    cudaDeviceProp dev{};
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&dev, deviceID);
    size_t totalMemory;
    size_t availMemory;
    cudaMemGetInfo(&availMemory, &totalMemory);
    long reduceMem = paramSize * sizeof(SIZE_TYPE) * (long) vertexArrSize;
    cout << "reduceMem " << reduceMem << " testNumNodes " << vertexArrSize << " ParamsSize " << paramSize
         << endl;
    total_gpu_size = (availMemory - reduceMem) / sizeof(EdgeType);

    //float adviseK = (10 - (float) edgeListSize / (float) totalSize) / 9;
    //uint dynamicDataMax = edgeListSize * edgeSize -
    float adviseK = (10 - (float) edgeArrSize / (float) total_gpu_size) / 9;
    cout << "adviseK " << adviseK << endl;
    if (adviseK < 0) {
        adviseK = 0.5;
        cout << "adviseK " << adviseK << endl;
    }
    if (adviseK > 1) {
        adviseK = 0.95;
        cout << "adviseK " << adviseK << endl;
    }
    float adviseRate = 0;
    if (adviseRate > 0) {
        adviseK = adviseRate;
    }

    max_partition_size = adviseK * total_gpu_size;
    cout << "availMemory " << availMemory << " totalMemory " << totalMemory << endl;
    printf("static memory is %ld totalGlobalMem is %ld, max static edge size is %ld\n total edge size %ld \n multiprocessors %d adviseK %f\n",
           availMemory - reduceMem,
           dev.totalGlobalMem, max_partition_size, total_gpu_size, dev.multiProcessorCount, adviseK);
    if (max_partition_size > UINT_MAX) {
        printf("bigger than DIST_INFINITY\n");
        max_partition_size = UINT_MAX;
    }
    SIZE_TYPE temp = max_partition_size % fragmentSize;
    max_partition_size = max_partition_size - temp;
    max_static_node = 0;
    SIZE_TYPE edgesInStatic = 0;
    for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
        if (nodePointers[i] < max_partition_size && (nodePointers[i] + degree[i] - 1) < max_partition_size) {
            isInStatic[i] = true;
            if (i > max_static_node) max_static_node = i;
            edgesInStatic += degree[i];
        } else {
            isInStatic[i] = false;
        }
    }

    partOverloadSize = total_gpu_size - max_partition_size;
    overloadSize = edgeArrSize - edgesInStatic;
}

template<class EdgeType>
void GraphMeta<EdgeType>::initLableAndValue() {

    label = new bool[vertexArrSize];
    if (algType == PR) {
        valuePr = new float[vertexArrSize];
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0;
        }
    } else {
        value = new SIZE_TYPE[vertexArrSize];
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }
    }
}

template<class EdgeType>
void GraphMeta<EdgeType>::refreshLabelAndValue() {
    if (algType == PR) {
        for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
            label[i] = 1;
            valuePr[i] = 1.0;
        }
        // todo
    } else {
        switch (algType) {
            case BFS:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case SSSP:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 0;
                    value[i] = UINT_MAX;
                }
                label[sourceNode] = 1;
                value[sourceNode] = 1;
                break;
            case CC:
                for (SIZE_TYPE i = 0; i < vertexArrSize; i++) {
                    label[i] = 1;
                    value[i] = i;
                }
        }

        cudaMemcpy(valueD, value, vertexArrSize * sizeof(SIZE_TYPE), cudaMemcpyHostToDevice);
        cudaMemcpy(isActiveD, label, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(isInStaticD, isInStatic, vertexArrSize * sizeof(bool), cudaMemcpyHostToDevice);
        gpuErrorcheck(cudaMemset(isStaticActive, 0, vertexArrSize * sizeof(bool)));
        gpuErrorcheck(cudaMemset(isOverloadActive, 0, vertexArrSize * sizeof(bool)));
    }
}
#endif //PTGRAPH_GRAPHMETA_CUH
