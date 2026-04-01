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
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF):-metal -compute -entry computeMain -allow-glsl -xslang -DMETAL

#version 430

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
buffer MyBlockName2 
{
    uint data[];
} outputBuffer;

layout(local_size_x = 4) in;

__generic<T : __BuiltinLogicalType>
bool test1QuadX() {
    return true
        & subgroupQuadSwapHorizontal(T(2)) == T(2)
        & subgroupQuadSwapVertical(T(2)) == T(2)
        & subgroupQuadSwapDiagonal(T(3)) == T(3)
        // subgroupQuadBroadcast is not implemented for WGSL and Metal as their intrinsics only accepts const integers expressions.
#if !defined(WGPU)
        & subgroupQuadBroadcast(T(1), 1) == T(1)
#endif
        ;
}
__generic<T : __BuiltinLogicalType, let N : int>
bool testVQuadX() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupQuadSwapHorizontal(gvec(T(2))) == gvec(T(2))
        & subgroupQuadSwapVertical(gvec(T(2))) == gvec(T(2))
        & subgroupQuadSwapDiagonal(gvec(T(3))) == gvec(T(3))
        // subgroupQuadBroadcast is not implemented for WGSL and Metal as their intrinsics only accepts const integers expressions.
#if !defined(WGPU)
        & subgroupQuadBroadcast(gvec(T(1)), 1) == gvec(T(1))
#endif
        ;
}

__generic<T : __BuiltinFloatingPointType>
bool test1QuadX() {
    return true
        & subgroupQuadSwapHorizontal(T(2)) == T(2)
        & subgroupQuadSwapVertical(T(2)) == T(2)
        & subgroupQuadSwapDiagonal(T(3)) == T(3)
        // subgroupQuadBroadcast is not implemented for WGSL and Metal as their intrinsics only accepts const integers expressions.
#if !defined(WGPU)
        & subgroupQuadBroadcast(T(1), 1) == T(1)
#endif
        ;
}
__generic<T : __BuiltinFloatingPointType, let N : int>
bool testVQuadX() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupQuadSwapHorizontal(gvec(T(2))) == gvec(T(2))
        & subgroupQuadSwapVertical(gvec(T(2))) == gvec(T(2))
        & subgroupQuadSwapDiagonal(gvec(T(3))) == gvec(T(3))
        // subgroupQuadBroadcast is not implemented for WGSL and Metal as their intrinsics only accepts const integers expressions.
#if !defined(WGPU)
        & subgroupQuadBroadcast(gvec(T(1)), 1) == gvec(T(1))
#endif
        ;
}
bool testQuadSwapX() {
    return true
        & test1QuadX<float>()
        & testVQuadX<float, 2>()
        & testVQuadX<float, 3>()
        & testVQuadX<float, 4>()
        & test1QuadX<half>() 
        & testVQuadX<half, 2>()
        & testVQuadX<half, 3>()
        & testVQuadX<half, 4>()
        & test1QuadX<int>()
        & testVQuadX<int, 2>()
        & testVQuadX<int, 3>()
        & testVQuadX<int, 4>()
        & test1QuadX<uint>()
        & testVQuadX<uint, 2>()
        & testVQuadX<uint, 3>()
        & testVQuadX<uint, 4>()

        // Disabled on WGSL and Metal as these built-in types are not supported.
#if !defined(WGPU) && !defined(METAL)
        & test1QuadX<double>() // WARNING: intel GPU's lack FP64 support
        & testVQuadX<double, 2>()
        & testVQuadX<double, 3>()
        & testVQuadX<double, 4>()
        & test1QuadX<int8_t>() 
        & testVQuadX<int8_t, 2>()
        & testVQuadX<int8_t, 3>()
        & testVQuadX<int8_t, 4>()
        & test1QuadX<int16_t>() 
        & testVQuadX<int16_t, 2>()
        & testVQuadX<int16_t, 3>()
        & testVQuadX<int16_t, 4>()
        & test1QuadX<int64_t>() 
        & testVQuadX<int64_t, 2>()
        & testVQuadX<int64_t, 3>()
        & testVQuadX<int64_t, 4>()
        & test1QuadX<uint8_t>() 
        & testVQuadX<uint8_t, 2>()
        & testVQuadX<uint8_t, 3>()
        & testVQuadX<uint8_t, 4>()
        & test1QuadX<uint64_t>() 
        & testVQuadX<uint64_t, 2>()
        & testVQuadX<uint64_t, 3>()
        & testVQuadX<uint64_t, 4>() & test1QuadX<bool>() & testVQuadX<bool, 2>()
        & testVQuadX<bool, 3>()
        & testVQuadX<bool, 4>()
#endif


#if !defined(WGPU)
        & test1QuadX<int16_t>() 
        & testVQuadX<int16_t, 2>()
        & testVQuadX<int16_t, 3>()
        & testVQuadX<int16_t, 4>()
        & test1QuadX<uint16_t>() 
        & testVQuadX<uint16_t, 2>()
        & testVQuadX<uint16_t, 3>()
        & testVQuadX<uint16_t, 4>()
#endif
        ;
}

void computeMain()
{

    outputBuffer.data[0] = true
        & testQuadSwapX()
        ;

    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(
    // BUF: 1
}
