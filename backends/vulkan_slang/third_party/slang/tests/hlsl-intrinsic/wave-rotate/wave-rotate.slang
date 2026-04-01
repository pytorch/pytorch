//TEST_CATEGORY(wave, compute)
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-directly
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-via-glsl
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-metal -compute -shaderobj -xslang -DMETAL


//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-directly -xslang -DUSE_GLSL_SYNTAX -allow-glsl
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-via-glsl -xslang -DUSE_GLSL_SYNTAX -allow-glsl
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-metal -compute -shaderobj -xslang -DMETAL -xslang -DUSE_GLSL_SYNTAX -allow-glsl


#if defined(USE_GLSL_SYNTAX)
#define __rotate subgroupRotate
#else
#define __rotate WaveRotate
#endif

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

#define SUBGROUP_SIZE 32
#define DELTA 3

static uint threadIndex;
static uint rotatedValue;

__generic<T : __BuiltinArithmeticType>
bool test1Rotate()
{
    return __rotate(T(threadIndex), DELTA) == T(rotatedValue);
}

__generic<T : __BuiltinArithmeticType, let N : int>
bool testVRotate()
{
    typealias gvec = vector<T, N>;

#if defined(USE_GLSL_SYNTAX)
    return (__rotate(gvec(T(threadIndex)), DELTA) == gvec(T(rotatedValue)));
#else
    return (__rotate(gvec(T(threadIndex)), DELTA) == gvec(T(rotatedValue)))[0];
#endif
}

bool test1RotateBool()
{
    bool currentValue = (threadIndex % 2 == 0) ? true : false;
    bool rotatedValueBool = (threadIndex % 2 == 0) ? false : true;
    return __rotate(currentValue, DELTA) == rotatedValueBool;
}

__generic<let N : int>
bool testVRotateBool()
{
    typealias gvec = vector<bool, N>;
    bool currentValue = (threadIndex % 2 == 0) ? true : false;
    bool rotatedValueBool = (threadIndex % 2 == 0) ? false : true;

#if defined(USE_GLSL_SYNTAX)
    return (__rotate(gvec(currentValue), DELTA) == gvec(rotatedValueBool));
#else
    return (__rotate(gvec(currentValue), DELTA) == gvec(rotatedValueBool))[0];
#endif
}

bool testRotate()
{
    return true
        & test1Rotate<float>()
        & testVRotate<float, 2>()
        & testVRotate<float, 3>()
        & testVRotate<float, 4>()
        & test1Rotate<half>()
        & testVRotate<half, 2>()
        & testVRotate<half, 3>()
        & testVRotate<half, 4>()
        & test1Rotate<uint>()
        & testVRotate<uint, 2>()
        & testVRotate<uint, 3>()
        & testVRotate<uint, 4>()
        & test1Rotate<uint16_t>()
        & testVRotate<uint16_t, 2>()
        & testVRotate<uint16_t, 3>()
        & testVRotate<uint16_t, 4>()
        & test1Rotate<int>()
        & testVRotate<int, 2>()
        & testVRotate<int, 3>()
        & testVRotate<int, 4>()
        & test1Rotate<int16_t>()
        & testVRotate<int16_t, 2>()
        & testVRotate<int16_t, 3>()
        & testVRotate<int16_t, 4>()

        // Subgroup rotate operations on these builtin types are not supported on Metal.
#if !defined(METAL)
        & test1Rotate<uint8_t>()
        & testVRotate<uint8_t, 2>()
        & testVRotate<uint8_t, 3>()
        & testVRotate<uint8_t, 4>()
        & test1Rotate<uint64_t>()
        & testVRotate<uint64_t, 2>()
        & testVRotate<uint64_t, 3>()
        & testVRotate<uint64_t, 4>()
        & test1Rotate<int8_t>()
        & testVRotate<int8_t, 2>()
        & testVRotate<int8_t, 3>()
        & testVRotate<int8_t, 4>()
        & test1Rotate<int64_t>()
        & testVRotate<int64_t, 2>()
        & testVRotate<int64_t, 3>()
        & testVRotate<int64_t, 4>()
        & test1RotateBool()
        & testVRotateBool<2>()
        & testVRotateBool<3>()
        & testVRotateBool<4>()
#endif
        ;
}

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void computeMain(uint3 dispatchID : SV_DispatchThreadID)
{
    threadIndex = dispatchID.x;
    rotatedValue = (threadIndex + DELTA) % SUBGROUP_SIZE;

    bool result = true
            & testRotate()
            ;

    // CHECK: 1
    outputBuffer[0] = uint(result);
}

