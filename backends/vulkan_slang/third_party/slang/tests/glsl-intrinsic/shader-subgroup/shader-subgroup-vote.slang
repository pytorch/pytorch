//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK_HLSL): -allow-glsl -stage compute -entry computeMain -target hlsl -DTARGET_HLSL

// not testing cuda due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CUDA): -allow-glsl -stage compute -entry computeMain -target cuda -DTARGET_CUDA 
// not testing cpp due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CPP):  -allow-glsl -stage compute -entry computeMain -target cpp -DTARGET_CPP

//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-wgpu -compute -entry computeMain -allow-glsl -xslang -DWGPU

// Not testing because CI runners may not support Metal's intrinsics.
//DISABLE_TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-metal -compute -entry computeMain -allow-glsl -xslang -DMETAL

#version 430

//TEST_INPUT:ubuffer(data=[9], stride=4):name=inputBuffer
buffer MyBlockName
{
    uint data[];
} inputBuffer;

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2 
{
    uint data[];
} outputBuffer;

layout(local_size_x = 32) in;

__generic<T : __BuiltinLogicalType>
bool test1AllEqual() {
    return true
        & subgroupAllEqual(T(1)) == true
        & subgroupAllEqual(T(gl_GlobalInvocationID.x)) == false
        ;
}
__generic<T : __BuiltinLogicalType, let N : int>
bool testVAllEqual() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupAllEqual(gvec(T(1))) == true
        & subgroupAllEqual(gvec(T(gl_GlobalInvocationID.x))) == false
        ;
}

__generic<T : __BuiltinFloatingPointType>
bool test1AllEqual() {
    return true
        & subgroupAllEqual(T(1)) == true
        & subgroupAllEqual(T(gl_GlobalInvocationID.x)) == false
        ;
}
__generic<T : __BuiltinFloatingPointType, let N : int>
bool testVAllEqual() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupAllEqual(gvec(T(1))) == true
        & subgroupAllEqual(gvec(T(gl_GlobalInvocationID.x))) == false
        ;
}
bool testAllEqual() {
    return true
        & test1AllEqual<float>()
        & testVAllEqual<float, 2>()
        & testVAllEqual<float, 3>()
        & testVAllEqual<float, 4>()
        & test1AllEqual<double>() // WARNING: intel GPU's lack FP64 support
        & testVAllEqual<double, 2>()
        & testVAllEqual<double, 3>()
        & testVAllEqual<double, 4>()
        & test1AllEqual<half>() 
        & testVAllEqual<half, 2>()
        & testVAllEqual<half, 3>()
        & testVAllEqual<half, 4>()
        & test1AllEqual<int>()
        & testVAllEqual<int, 2>()
        & testVAllEqual<int, 3>()
        & testVAllEqual<int, 4>()
        & test1AllEqual<int8_t>() 
        & testVAllEqual<int8_t, 2>()
        & testVAllEqual<int8_t, 3>()
        & testVAllEqual<int8_t, 4>()
        & test1AllEqual<int16_t>() 
        & testVAllEqual<int16_t, 2>()
        & testVAllEqual<int16_t, 3>()
        & testVAllEqual<int16_t, 4>()
        & test1AllEqual<int64_t>() 
        & testVAllEqual<int64_t, 2>()
        & testVAllEqual<int64_t, 3>()
        & testVAllEqual<int64_t, 4>()
        & test1AllEqual<uint>()
        & testVAllEqual<uint, 2>()
        & testVAllEqual<uint, 3>()
        & testVAllEqual<uint, 4>()
        & test1AllEqual<uint8_t>() 
        & testVAllEqual<uint8_t, 2>()
        & testVAllEqual<uint8_t, 3>()
        & testVAllEqual<uint8_t, 4>()
        & test1AllEqual<uint16_t>() 
        & testVAllEqual<uint16_t, 2>()
        & testVAllEqual<uint16_t, 3>()
        & testVAllEqual<uint16_t, 4>()
        & test1AllEqual<uint64_t>() 
        & testVAllEqual<uint64_t, 2>()
        & testVAllEqual<uint64_t, 3>()
        & testVAllEqual<uint64_t, 4>()
        & test1AllEqual<bool>()
        & testVAllEqual<bool, 2>()
        & testVAllEqual<bool, 3>()
        & testVAllEqual<bool, 4>()
        ;
}

[[ForceInline]]
void _barrier()
{
#if !defined(WGPU)
    subgroupBarrier();
#else
    GroupMemoryBarrier();
#endif
}

[shader("compute")]
void computeMain()
{
    //seperate tests since testing concurrency

    // one is true, rest false, positive
    outputBuffer.data[0] = 1;
    
#if !defined(WGPU)
    bool t1 = inputBuffer.data[0] == gl_GlobalInvocationID.x;
#else
    // There is no subgroup barrier for WGSL and workgroup barrier requries non uniform control flow.
    bool t1 = true;
#endif
    if (subgroupAny(t1)) {
        _barrier();
        outputBuffer.data[0] = 2;
    }

    // all false, negative
    outputBuffer.data[1] = 1;
    t1 = false;
    if (!subgroupAny(t1)) {
        _barrier();
        outputBuffer.data[1] = 2;
    }

    // all true, positive
    outputBuffer.data[2] = 1;
    t1 = true;
    if (subgroupAll(t1)) {
        _barrier();
        outputBuffer.data[2] = 2;
    }

    // all false, negative
    outputBuffer.data[3] = 1;
    t1 = false;
    if (!subgroupAll(t1)) {
        _barrier();
        outputBuffer.data[3] = 2;
    }

    outputBuffer.data[4] = 1;

    // All equal intrinsic is not supported on WGSL as of time of writing.
#if !defined(WGPU) && !defined(METAL)
    if (testAllEqual()) {
        subgroupBarrier();
        outputBuffer.data[4] = 2;
    }
#else
    outputBuffer.data[4] = 2;
#endif

    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(
    // BUF: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
    // BUF-NEXT: 2
}
