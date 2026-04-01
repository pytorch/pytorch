//TEST:SIMPLE(filecheck=CHECK_GLSL):  -allow-glsl -stage compute -entry computeMain -target glsl
//TEST:SIMPLE(filecheck=CHECK_SPV):  -allow-glsl -stage compute -entry computeMain -target spirv -emit-spirv-directly

// not testing hlsl due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_HLSL): -allow-glsl -stage compute -entry computeMain -target hlsl -DTARGET_HLSL
// not testing cuda due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CUDA): -allow-glsl -stage compute -entry computeMain -target cuda -DTARGET_CUDA 
// not testing cpp due to missing impl
//DISABLE_TEST:SIMPLE(filecheck=CHECK_CPP):  -allow-glsl -stage compute -entry computeMain -target cpp -DTARGET_CPP

//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl
//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -emit-spirv-directly
#version 430

//TEST_INPUT:ubuffer(data=[0 0], stride=4):out,name=outputBuffer
buffer MyBlockName2 
{
    uint data[];
} outputBuffer;

layout(local_size_x = 32) in;

__generic<T : __BuiltinLogicalType>
bool test1Logical() {
    return true
        & subgroupClusteredAnd(T(1), 1) == T(1)
        & subgroupClusteredOr(T(1), 1) == T(1)
        & subgroupClusteredXor(T(1), 1) == T(1)
        ;
}

__generic<T : __BuiltinLogicalType, let N : int>
bool testVLogical() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupClusteredAnd(gvec(T(1)), 1) == gvec(T(1))
        & subgroupClusteredOr(gvec(T(1)), 1) == gvec(T(1))
        & subgroupClusteredXor(gvec(T(1)), 1) == gvec(T(1))   
        ;     
}

bool testLogical() {
    return true
        & test1Logical<int>()
        & testVLogical<int, 2>()
        & testVLogical<int, 3>()
        & testVLogical<int, 4>()
        & test1Logical<int8_t>()
        & testVLogical<int8_t, 2>()
        & testVLogical<int8_t, 3>()
        & testVLogical<int8_t, 4>()
        & test1Logical<int16_t>()
        & testVLogical<int16_t, 2>()
        & testVLogical<int16_t, 3>()
        & testVLogical<int16_t, 4>()
        & test1Logical<int64_t>()
        & testVLogical<int64_t, 2>()
        & testVLogical<int64_t, 3>()
        & testVLogical<int64_t, 4>()
        & test1Logical<uint>()
        & testVLogical<uint, 2>()
        & testVLogical<uint, 3>()
        & testVLogical<uint, 4>()
        & test1Logical<uint8_t>()
        & testVLogical<uint8_t, 2>()
        & testVLogical<uint8_t, 3>()
        & testVLogical<uint8_t, 4>()
        & test1Logical<uint16_t>()
        & testVLogical<uint16_t, 2>()
        & testVLogical<uint16_t, 3>()
        & testVLogical<uint16_t, 4>()
        & test1Logical<uint64_t>()
        & testVLogical<uint64_t, 2>()
        & testVLogical<uint64_t, 3>()
        & testVLogical<uint64_t, 4>()
        & test1Logical<bool>()
        & testVLogical<bool, 2>()
        & testVLogical<bool, 3>()
        & testVLogical<bool, 4>()
        ;
}

__generic<T : __BuiltinArithmeticType>
bool test1Arithmetic() {
    return true
        & subgroupClusteredAdd(T(1), 1) == T(1)
        & subgroupClusteredMul(T(1), 1) == T(1)
        & subgroupClusteredMin(T(1), 1) == T(1)
        & subgroupClusteredMax(T(1), 1) == T(1)
        ;
}

__generic<T : __BuiltinArithmeticType, let N : int>
bool testVArithmetic() {
    typealias gvec = vector<T, N>;

    return true
        & subgroupClusteredAdd(gvec(T(1)), 1) == gvec(T(1))
        & subgroupClusteredMul(gvec(T(1)), 1) == gvec(T(1))
        & subgroupClusteredMin(gvec(T(1)), 1) == gvec(T(1))
        & subgroupClusteredMax(gvec(T(1)), 1) == gvec(T(1))
        ;
}

bool testArithmetic() {
    return true
        & test1Arithmetic<float>()
        & testVArithmetic<float, 2>()
        & testVArithmetic<float, 3>()
        & testVArithmetic<float, 4>()
        & test1Arithmetic<double>() // WARNING: intel GPU's lack FP64 support
        & testVArithmetic<double, 2>() 
        & testVArithmetic<double, 3>()
        & testVArithmetic<double, 4>()
        & test1Arithmetic<half>() 
        & testVArithmetic<half, 2>()
        & testVArithmetic<half, 3>()
        & testVArithmetic<half, 4>()
        & test1Arithmetic<int>()
        & testVArithmetic<int, 2>()
        & testVArithmetic<int, 3>()
        & testVArithmetic<int, 4>()
        & test1Arithmetic<int8_t>() 
        & testVArithmetic<int8_t, 2>()
        & testVArithmetic<int8_t, 3>()
        & testVArithmetic<int8_t, 4>()
        & test1Arithmetic<int16_t>() 
        & testVArithmetic<int16_t, 2>()
        & testVArithmetic<int16_t, 3>()
        & testVArithmetic<int16_t, 4>()
        & test1Arithmetic<int64_t>() 
        & testVArithmetic<int64_t, 2>()
        & testVArithmetic<int64_t, 3>()
        & testVArithmetic<int64_t, 4>()
        & test1Arithmetic<uint>()
        & testVArithmetic<uint, 2>()
        & testVArithmetic<uint, 3>()
        & testVArithmetic<uint, 4>()
        & test1Arithmetic<uint8_t>() 
        & testVArithmetic<uint8_t, 2>()
        & testVArithmetic<uint8_t, 3>()
        & testVArithmetic<uint8_t, 4>()
        & test1Arithmetic<uint16_t>() 
        & testVArithmetic<uint16_t, 2>()
        & testVArithmetic<uint16_t, 3>()
        & testVArithmetic<uint16_t, 4>()
        & test1Arithmetic<uint64_t>() 
        & testVArithmetic<uint64_t, 2>()
        & testVArithmetic<uint64_t, 3>()
        & testVArithmetic<uint64_t, 4>()
        ;
}

void computeMain()
{
    outputBuffer.data[0] = true
        & testLogical()
        ;
    outputBuffer.data[1] = true
        & testArithmetic()
        ;

    // CHECK_GLSL: void main(
    // CHECK_SPV: OpEntryPoint
    // CHECK_HLSL: void computeMain(
    // CHECK_CUDA: void computeMain(
    // CHECK_CPP: void _computeMain(
    // BUF: 1
    // BUF-NEXT: 1
}
