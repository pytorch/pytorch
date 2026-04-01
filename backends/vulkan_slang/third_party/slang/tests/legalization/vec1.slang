//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -output-using-type
//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj -emit-spirv-directly -output-using-type

// CHECK:      23
// CHECK-NEXT: 23
// CHECK-NEXT: 23
// CHECK-NEXT: 23

// This test tests that the 1-vector legalization works correctly.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// This struct helps test that nested access through 1-vectors works
struct V
{
    // 1-vector of 1-vector
    vector<vector<float, 1>, 1> oo;

    // 1-vector of n-vector
    vector<vector<float, 4>, 1> on;

    // n-vector of 1-vector
    vector<vector<float, 1>, 4> no;
};

vector<int, 1> get1Vec(int x)
{
    return x;
}

V getV()
{
    V v;

    // Test swizzle store
    v.oo.x.x = 1;

    // Test assigning into subscript
    v.on[0].wzyx = float4(4,3,2,1);

    // Test assigning from vector
    v.no.x = vector<float, 1>(1);

    // Test assigning from scalar
    v.no.y.x = 2;

    // Test assigning from vector of vector
    v.no.wz = vector<vector<float, 1>, 2>(3,4);

    return v;
}

float sumV(V v)
{
    return v.oo[0][0]
        + v.on.x.x
        + v.on.x.y
        + v.on.x.z
        + v.on.x.w
        // Test arithmetic
        + (v.no.x + v.no.y + v.no.z + v.no.w).x;
}

float3 splat(vector<float, 1> v)
{
    // Test swizzle
    return v.xxx;
}

// This function helps test that this legalization happens with generic length
// vectors specialized to 1
float triangle<let N : int>()
{
    vector<float, N> v;
    for(int i = 0; i < N; ++i)
        v[i] = i+1;

    float ret = 0;
    for(int i = 0; i < N; ++i)
        ret += v[i];
    return ret;
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const V v = getV();
    outputBuffer[dispatchThreadID.x]
        = sumV(v)
        + triangle<1>()
        + splat(v.oo.x).z;
}
