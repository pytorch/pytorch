//TEST_CATEGORY(wave, compute)
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-directly
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-via-glsl

//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-directly -xslang -DUSE_GLSL_SYNTAX -allow-glsl
//TEST:COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-vk -compute -shaderobj -emit-spirv-via-glsl -xslang -DUSE_GLSL_SYNTAX -allow-glsl

#if defined(USE_GLSL_SYNTAX)
#define __clusteredRotate subgroupClusteredRotate
#else
#define __clusteredRotate WaveClusteredRotate
#endif

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<uint> outputBuffer;

#define SUBGROUP_SIZE 32
#define DELTA 3
#define CLUSTER_SIZE 8

static uint threadIndex;
static uint clusterIndex;
static uint rotatedValue;

__generic<T : __BuiltinArithmeticType>
bool test1ClusteredRotate()
{
    return __clusteredRotate(T(threadIndex), DELTA, CLUSTER_SIZE) == T(rotatedValue);
}

__generic<T : __BuiltinArithmeticType, let N : int>
bool testVRClusteredRotate()
{
    typealias gvec = vector<T, N>;

#if defined(USE_GLSL_SYNTAX)
    return (__clusteredRotate(gvec(T(threadIndex)), DELTA, CLUSTER_SIZE) == gvec(T(rotatedValue)));
#else
    return (__clusteredRotate(gvec(T(threadIndex)), DELTA, CLUSTER_SIZE) == gvec(T(rotatedValue)))[0];
#endif
}

bool test1ClusteredRotateBool()
{
    bool currentValue = (threadIndex % 2 == 0) ? true : false;
    bool rotatedValueBool = (threadIndex % 2 == 0) ? false : true;
    return __clusteredRotate(currentValue, DELTA, CLUSTER_SIZE) == rotatedValueBool;
}

__generic<let N : int>
bool testVRClusteredRotateBool()
{
    typealias gvec = vector<bool, N>;
    bool currentValue = (threadIndex % 2 == 0) ? true : false;
    bool rotatedValueBool = (threadIndex % 2 == 0) ? false : true;

#if defined(USE_GLSL_SYNTAX)
    return (__clusteredRotate(gvec(currentValue), DELTA, CLUSTER_SIZE) == gvec(rotatedValueBool));
#else
    return (__clusteredRotate(gvec(currentValue), DELTA, CLUSTER_SIZE) == gvec(rotatedValueBool))[0];
#endif
}

bool testClusteredRotate()
{
    return true
        & test1ClusteredRotate<float>()
        & testVRClusteredRotate<float, 2>()
        & testVRClusteredRotate<float, 3>()
        & testVRClusteredRotate<float, 4>()
        & test1ClusteredRotate<half>()
        & testVRClusteredRotate<half, 2>()
        & testVRClusteredRotate<half, 3>()
        & testVRClusteredRotate<half, 4>()
        & test1ClusteredRotate<uint>()
        & testVRClusteredRotate<uint, 2>()
        & testVRClusteredRotate<uint, 3>()
        & testVRClusteredRotate<uint, 4>()
        & test1ClusteredRotate<uint16_t>()
        & testVRClusteredRotate<uint16_t, 2>()
        & testVRClusteredRotate<uint16_t, 3>()
        & testVRClusteredRotate<uint16_t, 4>()
        & test1ClusteredRotate<int>()
        & testVRClusteredRotate<int, 2>()
        & testVRClusteredRotate<int, 3>()
        & testVRClusteredRotate<int, 4>()
        & test1ClusteredRotate<int16_t>()
        & testVRClusteredRotate<int16_t, 2>()
        & testVRClusteredRotate<int16_t, 3>()
        & testVRClusteredRotate<int16_t, 4>()
        & test1ClusteredRotate<uint8_t>()
        & testVRClusteredRotate<uint8_t, 2>()
        & testVRClusteredRotate<uint8_t, 3>()
        & testVRClusteredRotate<uint8_t, 4>()
        & test1ClusteredRotate<uint64_t>()
        & testVRClusteredRotate<uint64_t, 2>()
        & testVRClusteredRotate<uint64_t, 3>()
        & testVRClusteredRotate<uint64_t, 4>()
        & test1ClusteredRotate<int8_t>()
        & testVRClusteredRotate<int8_t, 2>()
        & testVRClusteredRotate<int8_t, 3>()
        & testVRClusteredRotate<int8_t, 4>()
        & test1ClusteredRotate<int64_t>()
        & testVRClusteredRotate<int64_t, 2>()
        & testVRClusteredRotate<int64_t, 3>()
        & testVRClusteredRotate<int64_t, 4>()
        & test1ClusteredRotateBool()
        & testVRClusteredRotateBool<2>()
        & testVRClusteredRotateBool<3>()
        & testVRClusteredRotateBool<4>()
        ;
}

[shader("compute")]
[numthreads(SUBGROUP_SIZE, 1, 1)]
void computeMain(uint3 dispatchID : SV_DispatchThreadID)
{
    threadIndex = dispatchID.x;
    clusterIndex = dispatchID.x % CLUSTER_SIZE;

    // Determine expected value of clustered rotate in current invocation.
    // The values passed in are global invocation ids, and we rotate them withina cluster of size `CLUSTER_SIZE`.
    uint clusterStart = (threadIndex / CLUSTER_SIZE) * CLUSTER_SIZE;
    rotatedValue = clusterStart + ((threadIndex - clusterStart + DELTA) % CLUSTER_SIZE);

    bool result = true
            & testClusteredRotate()
            ;

    // CHECK: 1
    outputBuffer[0] = uint(result);
}

