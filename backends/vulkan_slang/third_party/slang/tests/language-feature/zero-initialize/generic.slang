//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(smoke,compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain -allow-glsl -profile sm_6_2 -xslang -zero-initialize -xslang -DDX12

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

__generic<T : __BuiltinArithmeticType>
bool getValue()
{
    T genericOut;
    return genericOut == T(0);
}
bool getValueBoolean()
{
    bool genericOut;
    return genericOut == bool(0);
}

__generic<T : __BuiltinArithmeticType, let N : int>
bool getValueVector()
{
    typealias gvec = vector<T, N>;

    gvec genericOut;
    for (int i = 0; i < N; ++i)
        if (genericOut[i] != T(0))
            return false;
    return true;
}
__generic<let N : int>
bool getValueVectorBoolean()
{
    typealias gvec = vector<bool, N>;

    gvec genericOut;
    for (int i = 0; i < N; ++i)
        if (genericOut[i] != bool(0))
            return false;
    return true;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
// BUF: 1
    outputBuffer[0] = true
    && getValue<int>()
#ifndef DX12
    && getValue<uint8_t>()
    && getValue<int8_t>()
#endif
    && getValue<uint16_t>()
    && getValue<int16_t>()
    && getValue<uint32_t>()
    && getValue<int32_t>()
    && getValue<uint64_t>()
    && getValue<int64_t>()
    && getValue<half>()
    && getValue<float>()
    && getValue<double>()
    && getValueBoolean()

    && getValueVector<int, 2>()
#ifndef DX12
    && getValueVector<uint8_t, 2>()
    && getValueVector<int8_t, 2>()
#endif
    && getValueVector<uint16_t, 2>()
    && getValueVector<int16_t, 2>()
    && getValueVector<uint32_t, 2>()
    && getValueVector<int32_t, 2>()
    && getValueVector<uint64_t, 2>()
    && getValueVector<int64_t, 2>()
    && getValueVector<half, 2>()
    && getValueVector<float, 2>()
    && getValueVector<double, 2>()
    && getValueVectorBoolean<2>()

    && getValueVector<int, 3>()
#ifndef DX12
    && getValueVector<uint8_t, 3>()
    && getValueVector<int8_t, 3>()
#endif
    && getValueVector<uint16_t, 3>()
    && getValueVector<int16_t, 3>()
    && getValueVector<uint32_t, 3>()
    && getValueVector<int32_t, 3>()
    && getValueVector<uint64_t, 3>()
    && getValueVector<int64_t, 3>()
    && getValueVector<half, 3>()
    && getValueVector<float, 3>()
    && getValueVector<double, 3>()
    && getValueVectorBoolean<3>()

    && getValueVector<int, 4>()
#ifndef DX12
    && getValueVector<uint8_t, 4>()
    && getValueVector<int8_t, 4>()
#endif
    && getValueVector<uint16_t, 4>()
    && getValueVector<int16_t, 4>()
    && getValueVector<uint32_t, 4>()
    && getValueVector<int32_t, 4>()
    && getValueVector<uint64_t, 4>()
    && getValueVector<int64_t, 4>()
    && getValueVector<half, 4>()
    && getValueVector<float, 4>()
    && getValueVector<double, 4>()
    && getValueVectorBoolean<4>()
    ;
}
