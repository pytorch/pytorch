//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-vk -compute -entry computeMain -emit-spirv-directly -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-cpu -compute -entry computeMain -allow-glsl -xslang -zero-initialize
//DISABLE_TEST(smoke,compute):COMPARE_COMPUTE_EX(filecheck-buffer=BUF):-dx12 -use-dxil -compute -entry computeMain -allow-glsl -profile sm_6_2 -xslang -zero-initialize -xslang -DDX12

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

enum MyFlags
{
    Zero = 0,
    One,
}
MyFlags getMyFlagsUint_One()
{
    return MyFlags::One;
}

#ifndef DX12
enum MyFlags_uint8 : uint8_t
{
    Zero = 0,
    One,
}
MyFlags_uint8 getMyFlagsUint8_One()
{
    return MyFlags_uint8::One;
}
#endif

enum MyFlags_uint16 : uint16_t
{
    Zero = 0,
    One,
}
MyFlags_uint16 getMyFlagsUint16_One()
{
    return MyFlags_uint16::One;
}

enum MyFlags_uint32 : uint32_t
{
    Zero = 0,
    One,
}
MyFlags_uint32 getMyFlagsUint32_One()
{
    return MyFlags_uint32::One;
}

enum MyFlags_uint64 : uint64_t
{
    Zero = 0,
    One,
}
MyFlags_uint64 getMyFlagsUint64_One()
{
    return MyFlags_uint64::One;
}

int getDefaultInt()
{
    int val;
    return val;
}
float getDefaultFloat()
{
    float val;
    return val;
}
double getDefaultDouble()
{
    double val;
    return val;
}
vector<int, 2> getDefaultIntVec2()
{
    vector<int, 2> val;
    return val;
}
vector<int, 3> getDefaultIntVec3()
{
    vector<int, 3> val;
    return val;
}
vector<int, 4> getDefaultIntVec4()
{
    vector<int, 4> val;
    return val;
}
[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
    int defaultInt = getDefaultInt();
    float defaultFloat = getDefaultFloat();
    double defaultDouble = getDefaultDouble();
    vector<int, 2> vector2 = getDefaultIntVec2();
    vector<int, 3> vector3 = getDefaultIntVec3();
    vector<int, 4> vector4 = getDefaultIntVec4();
// BUF: 1
    outputBuffer[0] = true
    && defaultInt == 0
    && defaultFloat == 0
    && defaultDouble == 0
    && vector2 == vector<int, 2>(0)
    && vector3 == vector<int, 3>(0)
    && vector4 == vector<int, 4>(0)
    && MyFlags::One == getMyFlagsUint_One()
#ifndef DX12
    && MyFlags_uint8::One == getMyFlagsUint8_One()
#endif
    && MyFlags_uint16::One == getMyFlagsUint16_One()
    && MyFlags_uint32::One == getMyFlagsUint32_One()
    && MyFlags_uint64::One == getMyFlagsUint64_One();
    ;
}
