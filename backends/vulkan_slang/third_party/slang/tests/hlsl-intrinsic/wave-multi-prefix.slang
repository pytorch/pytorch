//TEST:SIMPLE(filecheck=CHECK_SPIRV): -stage compute -entry computeMain -target spirv -DNO_INTEGER_MATRIX
//TEST:SIMPLE(filecheck=CHECK_GLSL): -stage compute -entry computeMain -target glsl -DNO_INTEGER_MATRIX
//TEST:SIMPLE(filecheck=CHECK_CUDA): -stage compute -entry computeMain -target cuda
//TEST:SIMPLE(filecheck=CHECK_HLSL): -stage compute -entry computeMain -target hlsl

//
// Tests all variants and overloads of WaveMultiPrefix* arithmetic intrinsics.
//

struct OutputData
{
    int scalarSum;
    int scalarProduct;
    int scalarBitAnd;
    int scalarBitOr;
    int scalarBitXor;
    int vectorSum;
    int vectorProduct;
    int vectorBitAnd;
    int vectorBitOr;
    int vectorBitXor;
    int matrixSum;
    int matrixProduct;
    int matrixBitAnd;
    int matrixBitOr;
    int matrixBitXor;
    float floatScalarSum;
    float floatScalarProduct;
    float floatVectorSum;
    float floatVectorProduct;
    float floatMatrixSum;
    float floatMatrixProduct;
};

RWStructuredBuffer<OutputData> outputBuffer;

// CHECK_SPIRV: OpCapability GroupNonUniformPartitionedNV
// CHECK_SPIRV: OpExtension "SPV_NV_shader_subgroup_partitioned"
// CHECK_SPIRV: OpGroupNonUniformIAdd{{.*}}PartitionedExclusiveScanNV
// CHECK_SPIRV: OpGroupNonUniformIMul{{.*}}PartitionedExclusiveScanNV
// CHECK_SPIRV: OpGroupNonUniformBitwiseAnd{{.*}}PartitionedExclusiveScanNV
// CHECK_SPIRV: OpGroupNonUniformBitwiseOr{{.*}}PartitionedExclusiveScanNV
// CHECK_SPIRV: OpGroupNonUniformBitwiseXor{{.*}}PartitionedExclusiveScanNV
// CHECK_SPIRV: OpGroupNonUniformFAdd{{.*}}PartitionedExclusiveScanNV

// CHECK_GLSL: GL_NV_shader_subgroup_partitioned
// CHECK_GLSL: subgroupPartitionedExclusiveAddNV
// CHECK_GLSL: subgroupPartitionedExclusiveMulNV
// CHECK_GLSL: subgroupPartitionedExclusiveAndNV
// CHECK_GLSL: subgroupPartitionedExclusiveOrNV
// CHECK_GLSL: subgroupPartitionedExclusiveXorNV

// CHECK_CUDA: _wavePrefixSum
// CHECK_CUDA: _wavePrefixProduct
// CHECK_CUDA: _wavePrefixAnd
// CHECK_CUDA: _wavePrefixOr
// CHECK_CUDA: _wavePrefixXor
// CHECK_CUDA: _wavePrefixSumMultiple
// CHECK_CUDA: _wavePrefixProductMultiple
// CHECK_CUDA: _wavePrefixAndMultiple
// CHECK_CUDA: _wavePrefixOrMultiple
// CHECK_CUDA: _wavePrefixXorMultiple

// CHECK_HLSL: WaveMultiPrefixSum
// CHECK_HLSL: WaveMultiPrefixProduct
// CHECK_HLSL: WaveMultiPrefixBitAnd
// CHECK_HLSL: WaveMultiPrefixBitOr
// CHECK_HLSL: WaveMultiPrefixBitXor


[numthreads(1, 1, 1)]
void computeMain(uint3 dTid : SV_DispatchThreadID)
{
    int scalarVal = dTid.x;
    uint4 mask = WaveMatch(scalarVal);

    int scalarSum = WaveMultiPrefixSum(scalarVal, mask);
    int scalarProduct = WaveMultiPrefixProduct(scalarVal, mask);
    int scalarBitAnd = WaveMultiPrefixBitAnd(scalarVal, mask);
    int scalarBitOr = WaveMultiPrefixBitOr(scalarVal, mask);
    int scalarBitXor = WaveMultiPrefixBitXor(scalarVal, mask);

    int3 vectorVal = int3(dTid.x, dTid.y, dTid.z);
    int3 vectorSum = WaveMultiPrefixSum(vectorVal, mask);
    int3 vectorProduct = WaveMultiPrefixProduct(vectorVal, mask);
    int3 vectorBitAnd = WaveMultiPrefixBitAnd(vectorVal, mask);
    int3 vectorBitOr = WaveMultiPrefixBitOr(vectorVal, mask);
    int3 vectorBitXor = WaveMultiPrefixBitXor(vectorVal, mask);

    float floatScalarVal = float(dTid.x) + 0.5f; // Example floating-point scalar value
    uint4 floatMask = WaveMatch(floatScalarVal); // Create a mask for matching lanes

    float floatScalarSum = WaveMultiPrefixSum(floatScalarVal, floatMask);
    float floatScalarProduct = WaveMultiPrefixProduct(floatScalarVal, floatMask);

    float3 floatVectorVal = float3(dTid.x, dTid.y, dTid.z) + 0.5f; // Example floating-point vector value
    float3 floatVectorSum = WaveMultiPrefixSum(floatVectorVal, floatMask);
    float3 floatVectorProduct = WaveMultiPrefixProduct(floatVectorVal, floatMask);

    OutputData output;
    output.scalarSum = scalarSum;
    output.scalarProduct = scalarProduct;
    output.scalarBitAnd = scalarBitAnd;
    output.scalarBitOr = scalarBitOr;
    output.scalarBitXor = scalarBitXor;
    output.vectorSum = vectorSum.x;
    output.vectorProduct = vectorProduct.x;
    output.vectorBitAnd = vectorBitAnd.x;
    output.vectorBitOr = vectorBitOr.x;
    output.vectorBitXor = vectorBitXor.x;
    output.floatScalarSum = floatScalarSum;
    output.floatScalarProduct = floatScalarProduct;
    output.floatVectorSum = floatVectorSum.x;
    output.floatVectorProduct = floatVectorProduct.x;

    float3x3 floatMatrixVal = float3x3(
        float(dTid.x) + 0.5f, float(dTid.y) + 0.5f, float(dTid.z) + 0.5f,
        float(dTid.z) + 0.5f, float(dTid.x) + 0.5f, float(dTid.y) + 0.5f,
        float(dTid.y) + 0.5f, float(dTid.z) + 0.5f, float(dTid.x) + 0.5f
    );
    float3x3 floatMatrixSum = WaveMultiPrefixSum(floatMatrixVal, floatMask);
    float3x3 floatMatrixProduct = WaveMultiPrefixProduct(floatMatrixVal, floatMask);
    output.floatMatrixSum = floatMatrixSum[0][0];
    output.floatMatrixProduct = floatMatrixProduct[0][0];

#if !defined(NO_INTEGER_MATRIX)
    int3x3 matrixVal = int3x3(
        dTid.x, dTid.y, dTid.z,
        dTid.z, dTid.x, dTid.y,
        dTid.y, dTid.z, dTid.x
    );
    int3x3 matrixSum = WaveMultiPrefixSum(matrixVal, mask);
    int3x3 matrixProduct = WaveMultiPrefixProduct(matrixVal, mask);
    int3x3 matrixBitAnd = WaveMultiPrefixBitAnd(matrixVal, mask);
    int3x3 matrixBitOr = WaveMultiPrefixBitOr(matrixVal, mask);
    int3x3 matrixBitXor = WaveMultiPrefixBitXor(matrixVal, mask);
    output.matrixSum = matrixSum[0][0];
    output.matrixProduct = matrixProduct[0][0];
    output.matrixBitAnd = matrixBitAnd[0][0];
    output.matrixBitOr = matrixBitOr[0][0];
    output.matrixBitXor = matrixBitXor[0][0];
#endif

    outputBuffer[dTid.x] = output;
}

