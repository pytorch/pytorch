//IGNORE_TEST:CPU_REFLECTION: -profile cs_5_0 -entry computeMain -target cpp
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-cpu -compute
//TEST:SIMPLE(filecheck=DXIL):-target dxbc-assembly -entry computeMain -profile cs_5_1
//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry computeMain -profile cs_5_1 -emit-spirv-via-glsl
//DISABLED_TEST(compute):COMPARE_COMPUTE_EX:-cuda -compute

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

//TEST_INPUT:array(size=2):name g_aoa
RWStructuredBuffer<int> g_aoa[];

//TEST_INPUT:ubuffer(data=[1 2 3 4], stride=4):name=g_aoa[0]
//TEST_INPUT:ubuffer(data=[8 17 34], stride=4):name=g_aoa[1]

[numthreads(8, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // DXIL: cs_5_1
    // CHECK: OpCapability {{(ShaderNonUniform|StorageBufferArrayNonUniformIndexing)}}
    // CHECK: OpCapability {{(ShaderNonUniform|StorageBufferArrayNonUniformIndexing)}}
    // CHECK-DAG: OpDecorate %[[N1:[a-zA-Z0-9_]+]] NonUniform
    // CHECK-DAG: OpDecorate %[[N2:[a-zA-Z0-9_]+]] NonUniform
    // CHECK-DAG: OpDecorate %[[N3:[a-zA-Z0-9_]+]] NonUniform
    // CHECK-DAG: OpDecorate %[[N4:[a-zA-Z0-9_]+]] NonUniform
    // CHECK-DAG: OpDecorate %[[B0:[a-zA-Z0-9_]+]] Binding 0
    // CHECK-DAG: OpDecorate %[[B1:[a-zA-Z0-9_]+]] Binding 1
    // CHECK-DAG: %[[P1:[a-zA-Z0-9_]+]] = OpTypePointer Uniform %int
    // CHECK-DAG: %[[P2:[a-zA-Z0-9_]+]] = OpTypePointer Uniform %StructuredBuffer_int_t_0
    // CHECK: %[[N1]] = OpCopyObject %int
    // CHECK: %{{.*}} = OpAccessChain %[[P2]] %[[B1]] %[[N1]]
    // CHECK: %[[N2]] = OpCopyObject %int
    // CHECK: %[[N3]] = OpAccessChain %[[P1]] %[[B1]] %[[N2]]
    // CHECK: %[[N4]] = OpLoad %int %[[N3]]
    // CHECK: %{{.*}} = OpAccessChain %[[P1]] %[[B0]] %int_{{.*}}
    // CHECK: OpStore %{{.*}} %[[N4]]
    int index = int(dispatchThreadID.x);
    
    int baseIndex = index >> 2;
    int innerIndex = index & 3;

    RWStructuredBuffer<int> buffer = g_aoa[NonUniformResourceIndex(baseIndex)];

    // Get the size 
    uint bufferCount, bufferStride;
    buffer.GetDimensions(bufferCount, bufferStride);

    if (innerIndex >= bufferCount)
    {
        innerIndex = int(bufferCount - 1);
    }
	outputBuffer[index] = buffer[innerIndex]; 
}