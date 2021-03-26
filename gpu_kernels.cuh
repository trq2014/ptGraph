
#include "range.hpp"
#include "globals.cuh"
using namespace util::lang;

// type alias to simplify typing...
template<typename T>
using step_range = typename range_proxy<T>::step_range_proxy;

template<typename T>
__device__
step_range<T> grid_stride_range(T begin, T end) {
    begin += blockDim.x * blockIdx.x + threadIdx.x;
    return range(begin, end).step(gridDim.x * blockDim.x);
}

template<typename Predicate>
__device__ void streamVertices(int vertices, Predicate p) {
    for (auto i : grid_stride_range(0, vertices)) {
        p(i);
    }
}

uint reduceBool(uint* resultD, bool* isActiveD, uint vertexSize, dim3 grid, dim3 block);
__device__ void reduceStreamVertices(int vertices, bool *rawData, uint *result);
__global__ void reduceByBool(uint vertexSize, bool *rawData, uint *result);
template <int blockSize> __global__ void reduceResult(uint *result);

__global__ void
bfs_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
           bool *labelD);

__global__ void
cc_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
          bool *labelD);

__global__ void
sssp_kernel(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
            uint *valueD,
            bool *labelD);

__global__ void
bfs_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
              bool *labelD, uint overloadNode, uint *overloadEdgeListD,
              uint *nodePointersOverloadD);

__global__ void
bfs_kernelStatic2Label(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                       uint *valueD,
                       uint *isActiveD1, uint *isActiveD2);

__global__ void
bfs_kernelDynamic2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                        const uint *degreeD,
                        uint *valueD,
                        uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                        const uint *activeOverloadNodePointersD);

__global__ void
bfs_kernelDynamicPart(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                      const uint *degreeD,
                      uint *valueD,
                      uint *isActiveD, const uint *edgeListOverloadD,
                      const uint *activeOverloadNodePointersD);

__global__ void
setStaticAndOverloadNodePointer(uint vertexNum, uint *staticNodes, uint *overloadNodes, uint *overloadNodePointers,
                                uint *staticLabel, uint *overloadLabel,
                                uint *staticPrefix, uint *overloadPrefix, uint *degreeD);

__global__ void
sssp_kernelStaticSwapOpt2Label(uint activeNodesNum, const uint *activeNodeListD,
                               const uint *staticNodePointerD, const uint *degreeD,
                               EdgeWithWeight *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2,
                               bool *isFinish);

__global__ void
sssp_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                             const uint *degreeD,
                             uint *valueD,
                             uint *isActiveD1, uint *isActiveD2, const EdgeWithWeight *edgeListOverloadD,
                             const uint *activeOverloadNodePointersD, bool *finished);

__global__ void
bfs_kernelStatic(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
                 uint *labelD);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, bool *isInD, uint *fragmentRecordsD, uint fragment_size);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, uint *fragmentRecordsD, uint fragment_size);

__global__ void
bfs_kernelStaticSwap(uint nodeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                     uint *valueD,
                     uint *labelD, uint *fragmentRecordsD, uint fragment_size, uint maxpartionSize, uint testNumNodes);

__global__ void
bfs_kernelDynamic(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                  uint *labelD, uint overloadNode, uint *overloadEdgeListD,
                  uint *nodePointersOverloadD);

__global__ void
bfs_kernelDynamicSwap(uint activeNum, uint *activeNodesD, uint *degreeD, uint *valueD,
                      uint *labelD, uint *overloadEdgeListD,
                      uint *nodePointersOverloadD);

__global__ void
sssp_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
               uint *valueD,
               bool *labelD, uint overloadNode, EdgeWithWeight *overloadEdgeListD, uint *nodePointersOverloadD);
__global__ void
sssp_kernelDynamicUvm(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, EdgeWithWeight *edgeListD,
                      uint *valueD,
                      uint *labelD1, uint *labelD2);

__global__ void
cc_kernelOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD, uint *valueD,
             bool *labelD, uint overloadNode, uint *overloadEdgeListD, uint *nodePointersOverloadD);

__global__ void
cc_kernelDynamicSwap(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                     uint *valueD,
                     uint *isActiveD, const uint *edgeListOverloadD,
                     const uint *activeOverloadNodePointersD);

__global__ void
cc_kernelDynamicSwap2Label(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                           const uint *degreeD,
                           uint *valueD,
                           uint *isActiveD1, uint *isActiveD2, const uint *edgeListOverloadD,
                           const uint *activeOverloadNodePointersD, bool *finished);

__global__ void
cc_kernelDynamicAsync(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD, const uint *degreeD,
                      uint *valueD, uint *labelD1, uint *labelD2, const uint *edgeListOverloadD,
                      const uint *activeOverloadNodePointersD, bool *finished);

__global__ void
cc_kernelStaticSwap(uint activeNodesNum, uint *activeNodeListD,
                    uint *staticNodePointerD, uint *degreeD,
                    uint *edgeListD, uint *valueD, uint *isActiveD, bool *isInStaticD);

__global__ void
cc_kernelStaticAsync(uint activeNodesNum, const uint *activeNodeListD,
                     const uint *staticNodePointerD, const uint *degreeD,
                     const uint *edgeListD, uint *valueD, uint *labelD1, uint *labelD2, const bool *isInStaticD,
                     bool *finished, int *atomicValue);

__global__ void
bfs_kernelOptOfSorted(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                      uint *edgeListOverload, uint *valueD, bool *labelD, bool *isInListD, uint *nodePointersOverloadD);

__global__ void
bfs_kernelShareOpt(uint activeNum, uint *activeNodesD, uint *nodePointersD, uint *degreeD, uint *edgeListD,
                   uint *edgeListShare, uint *valueD, bool *labelD, uint overloadNode);

__global__ void
setLabelDefault(uint activeNum, uint *activeNodes, bool *labelD);

__global__ void
setLabelDefaultOpt(uint activeNum, uint *activeNodes, uint *labelD);

__global__ void
mixStaticLabel(uint activeNum, uint *activeNodes, uint *labelD1, uint *labelD2, bool *isInD);

__global__ void
mixDynamicPartLabel(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1, uint *labelD2);

__global__ void
setDynamicPartLabelTrue(uint overloadPartNodeNum, uint startIndex, const uint *overloadNodes, uint *labelD1,
                        uint *labelD2);

__global__ void
mixCommonLabel(uint testNodeNum, uint *labelD1, uint *labelD2);

__global__ void
cleanStaticAndOverloadLabel(uint vertexNum, uint *staticLabel, uint *overloadLabel);

__global__ void
setStaticAndOverloadLabel(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabelBool(uint vertexNum, bool *activeLabel, bool *staticLabel, bool *overloadLabel, bool *isInD);

__global__ void
setStaticAndOverloadLabel4Pr(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel, bool *isInD,
                             uint *fragmentRecordD, uint *nodePointersD, uint fragment_size, uint *degreeD,
                             bool *isFragmentActiveD);

__global__ void
setOverloadActiveNodeArray(uint vertexNum, uint *activeNodes, uint *overloadLabel,
                           uint *activeLabelPrefix);

__global__ void
setStaticActiveNodeArray(uint vertexNum, uint *activeNodes, uint *staticLabel,
                         uint *activeLabelPrefix);

__global__ void
cc_kernelStaticSwapOpt(uint activeNodesNum, uint *activeNodeListD,
                       uint *staticNodePointerD, uint *degreeD,
                       uint *edgeListD, uint *valueD, uint *isActiveD);

__global__ void
cc_kernelStaticSwapOpt2Label(uint activeNodesNum, uint *activeNodeListD,
                             uint *staticNodePointerD, uint *degreeD,
                             uint *edgeListD, uint *valueD, uint *isActiveD1, uint *isActiveD2, bool *isFinish);

__global__ void
setLabeling(uint vertexNum, bool *labelD, uint *labelingD);

__global__ void
setActiveNodeArray(uint vertexNum, uint *activeNodes, bool *activeLabel, uint *activeLabelPrefix);

__global__ void
setActiveNodeArrayAndNodePointer(uint vertexNum, uint *activeNodes, uint *activeNodePointers, bool *activeLabel,
                                 uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerBySortOpt(uint vertexNum, uint *activeNodes, uint *activeOverloadDegree,
                                          bool *activeLabel, uint *activeLabelPrefix, bool *isInList, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerOpt(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                                    uint *activeLabelPrefix, uint overloadVertex, uint *degreeD);

__global__ void
setActiveNodeArrayAndNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeLabel,
                                     uint *activeLabelPrefix, bool *isInD);

__global__ void
setOverloadNodePointerSwap(uint vertexNum, uint *activeNodes, uint *activeNodePointers, uint *activeLabel,
                           uint *activeLabelPrefix, uint *degreeD);

__global__ void
setFragmentData(uint activeNodeNum, uint *activeNodeList, uint *staticNodePointers, uint *staticFragmentData,
                uint staticFragmentNum, uint fragmentSize, bool *isInStatic);

__global__ void
setStaticFragmentData(uint staticFragmentNum, uint *canSwapFragmentD, uint *canSwapFragmentPrefixD,
                      uint *staticFragmentDataD);

__global__ void
setFragmentDataOpt(uint *staticFragmentData, uint staticFragmentNum, uint *staticFragmentVisitRecordsD);

__global__ void
recordFragmentVisit(uint *activeNodeListD, uint activeNodeNum, uint *nodePointersD, uint *degreeD, uint fragment_size,
                    uint *fragmentRecordsD);

__global__ void
bfsKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                          const uint *nodePointersD,
                          const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD);

__global__ void
prSumKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                            const uint *nodePointersD,
                            const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const float *valueD,
                            float *sumD);

__global__ void
prKernel_CommonPartition(uint nodeNum, float *valueD, float *sumD, bool *isActiveNodeList);

__global__ void
prSumKernel_UVM(uint vertexNum, const int *isActiveNodeListD, const uint *nodePointersD,
                const uint *edgeListD, const uint *degreeD, const float *valueD, float *sumD);

__global__ void
prKernel_UVM(uint nodeNum, float *valueD, float *sumD, int *isActiveListD);

__global__ void
prSumKernel_UVM_Out(uint vertexNum, int *isActiveNodeListD, const uint *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, float *valueD);

__global__ void
prKernel_UVM_outDegree(uint nodeNum, float *valueD, float *sumD, int *isActiveListD);

__global__ void
prSumKernel_static(uint activeNum, const uint *activeNodeList,
                   const uint *nodePointersD,
                   const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const float *valueD,
                   float *sumD);

__global__ void
prSumKernel_dynamic(uint overloadStartNode, uint overloadNodeNum, const uint *overloadNodeListD,
                    const uint *nodePointersD,
                    const uint *edgeListD, const uint *degreeD, const uint *outDegreeD, const float *valueD,
                    float *sumD);

__global__ void prKernel_Opt(uint nodeNum, float *valueD, float *sumD, uint *isActiveNodeList);

__global__ void
setFragmentDataOpt4Pr(uint *staticFragmentData, uint fragmentNum, uint *fragmentVisitRecordsD,
                      bool *isActiveFragmentD, uint* fragmentNormalMap2StaticD, uint maxStaticFragment);

__global__ void
ccKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                         const uint *nodePointersD,
                         const uint *edgeListD, const uint *degreeD, uint *valueD, bool *nextActiveNodeListD);

__global__ void
ssspKernel_CommonPartition(uint startVertex, uint endVertex, uint offset, const bool *isActiveNodeListD,
                           const uint *nodePointersD,
                           const EdgeWithWeight *edgeListD, const uint *degreeD, uint *valueD,
                           bool *nextActiveNodeListD);
__global__ void
setStaticAndOverloadLabelAndRecord(uint vertexNum, uint *activeLabel, uint *staticLabel, uint *overloadLabel,
                                   bool *isInD, uint *vertexVisitRecordD);

template<int NT>
__device__ int reduceInWarp(int idInWarp, bool data);