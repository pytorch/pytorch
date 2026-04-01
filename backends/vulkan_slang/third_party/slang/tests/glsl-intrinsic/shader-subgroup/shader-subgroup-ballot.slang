//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK_HLSL): -allow-glsl -stage compute -entry computeMain -target hlsl -DTARGET_HLSL

// not testing cuda due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CUDA): -allow-glsl -stage compute -entry computeMain -target cuda -DTARGET_CUDA 
// not testing cpp due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CPP):  -allow-glsl -stage compute -entry computeMain -target cpp -DTARGET_CPP

//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-wgpu -compute -entry computeMain -allow-glsl -xslang -DWGPU -render-feature half

// Not testing because CI runners may not support Metal's intrinsics.
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-metal -compute -entry computeMain -allow-glsl -xslang -DMETAL

#version 430

// breaks on Nvidia GPU by returning 0 which is trivially wrong (works on Intel Iris Xe)
//#define TEST_when_glsl_subgroupBallotExclusiveBitCount_is_not_bugged

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2 
{
    uint data[];
} outputBuffer;

layout(local_size_x = 32) in;

__generic<T : __BuiltinLogicalType>
bool test1BroadcastX() {
    return true
        & subgroupBroadcast(T(1), 0) == T(1)
        & subgroupBroadcastFirst(T(1)) == T(1)
        ;
}
__generic<T : __BuiltinLogicalType, let N : int>
bool testVBroadcastX() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupBroadcast(gvec(T(1)), 0) == gvec(T(1))
        & subgroupBroadcastFirst(gvec(T(1))) == gvec(T(1))
        ;
}

__generic<T : __BuiltinFloatingPointType>
bool test1BroadcastX() {
    return true
        & subgroupBroadcast(T(1), 0) == T(1)
        & subgroupBroadcastFirst(T(1)) == T(1)
        ;
}
__generic<T : __BuiltinFloatingPointType, let N : int>
bool testVBroadcastX() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupBroadcast(gvec(T(1)), 0) == gvec(T(1))
        & subgroupBroadcastFirst(gvec(T(1))) == gvec(T(1))
        ;
}
bool testBroadcastX() {
    return true
        & test1BroadcastX<float>()
        & testVBroadcastX<float, 2>()
        & testVBroadcastX<float, 3>()
        & testVBroadcastX<float, 4>()
        & test1BroadcastX<half>() 
        & testVBroadcastX<half, 2>()
        & testVBroadcastX<half, 3>()
        & testVBroadcastX<half, 4>()
        & test1BroadcastX<int>()
        & testVBroadcastX<int, 2>()
        & testVBroadcastX<int, 3>()
        & testVBroadcastX<int, 4>()
        & test1BroadcastX<uint>()
        & testVBroadcastX<uint, 2>()
        & testVBroadcastX<uint, 3>()
        & testVBroadcastX<uint, 4>()

        // Disabled on WGPU and Metal as these built-in types are not supported as of time of writing.
#if !defined(WGPU) && !defined(METAL)
        & test1BroadcastX<double>() // WARNING: intel GPU's lack FP64 support
        & testVBroadcastX<double, 2>()
        & testVBroadcastX<double, 3>()
        & testVBroadcastX<double, 4>()
        & test1BroadcastX<int8_t>() 
        & testVBroadcastX<int8_t, 2>()
        & testVBroadcastX<int8_t, 3>()
        & testVBroadcastX<int8_t, 4>()
        & test1BroadcastX<int64_t>() 
        & testVBroadcastX<int64_t, 2>()
        & testVBroadcastX<int64_t, 3>()
        & testVBroadcastX<int64_t, 4>()
        & test1BroadcastX<uint8_t>() 
        & testVBroadcastX<uint8_t, 2>()
        & testVBroadcastX<uint8_t, 3>()
        & testVBroadcastX<uint8_t, 4>()
        & test1BroadcastX<uint64_t>() 
        & testVBroadcastX<uint64_t, 2>()
        & testVBroadcastX<uint64_t, 3>()
        & testVBroadcastX<uint64_t, 4>()
        & test1BroadcastX<bool>()
        & testVBroadcastX<bool, 2>()
        & testVBroadcastX<bool, 3>()
        & testVBroadcastX<bool, 4>()
#endif

        // Disabled on WGPU as these built-in types are not supported as of time of writing.
#if !defined(WGPU)
        & test1BroadcastX<int16_t>() 
        & testVBroadcastX<int16_t, 2>()
        & testVBroadcastX<int16_t, 3>()
        & testVBroadcastX<int16_t, 4>()
        & test1BroadcastX<uint16_t>() 
        & testVBroadcastX<uint16_t, 2>()
        & testVBroadcastX<uint16_t, 3>()
        & testVBroadcastX<uint16_t, 4>()
#endif
        ;
}

bool testBallot() {
    return true 
        & (subgroupBallot(true).x == 0xFFFFFFFF)

#if !defined(WGPU) && !defined(METAL)
        & (subgroupInverseBallot(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) == true)
        & (subgroupBallotBitExtract(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF), 0) == true)
        & (subgroupBallotBitCount(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) == 32)
        & (subgroupBallotInclusiveBitCount(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) != 0)
#ifdef TEST_when_glsl_subgroupBallotExclusiveBitCount_is_not_bugged
        & (subgroupBallotExclusiveBitCount(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) != 0)
#endif
        & (subgroupBallotFindLSB(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) == 0)
        & (subgroupBallotFindMSB(uvec4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)) == 31)
#endif
        ;
}

[shader("compute")]
void computeMain()
{
    outputBuffer.data[0] = true
        & testBroadcastX()
        ;
    outputBuffer.data[1] = true
        & testBallot()
        ;

    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(
    // BUF: 1
    // BUF-NEXT: 1
}
