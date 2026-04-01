//TEST(compute):COMPARE_COMPUTE:-cpu
//TEST(compute):COMPARE_COMPUTE:-cuda

enum Color
{
    Red,
    Green = 2,
    Blue,
}

int test(int val)
{
    Color c = Color.Red;

    if(val > 1)
    {
        c = Color.Green;
    }

    if(c == Color.Red)
    {
        if((val & 1) != 0)
        {
            c = Color.Blue;
        }
    }

    switch(c)
    {
    case Color.Red:
        val = 1;
        break;

    case Color.Green:
        val = 2;
        break;

    case Color.Blue:
        val = 3;
        break;

    default:
        val = -1;
        break;
    }

    return (val << 4) + int(c);
}

float sum(float a[3])
{
    float total = a[0];
    for (int i = 1; i < 3; ++i)
    {
        total += a[i];
    }
    return total;
}

struct Thing
{
    int a;
    float b;
};

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint tid = dispatchThreadID.x;

    Thing thing = { 10, -1.0 };

    float array[3] = { thing.a, 2, 3};

    float anotherArray[] = { 1, 2, 5 };

    array[0] += anotherArray[1];

    matrix<float, 2, 3> mat = { { sum(array), 1, 2 }, { 3, 4, 5} };
    vector<float, 2> vec = { float(tid + 1), float(tid + 2) };

    vec += float2(7, 11);

    vector<float, 3> vec2 = max(sin(mul(vec, mat)), float3(1, 2, -1));
    vector<float, 3> vec3 = mul(vec, mat);
    
    float3 vec4 = lerp(vec2, vec3, float3(tid * (1.0f / 4), 1, 1));
    
    float3 crossVec = normalize(cross(vec4, vec4) + float3(2, 3, 1));
    
    vec2.x = fmod(crossVec.y, crossVec.x);
    
    vec2 = fmod(vec2, crossVec);
    
    vec2 += (-vec2.zyx) * 2 + crossVec * length(crossVec) + reflect(vec4, normalize(crossVec));
    
    vector<bool, 3> z = vec2 > 0;

    int val = (int(tid) + (any(z) ? 1 : 0) + (all(z) ? 2 : 0)) % 100;
    
    val = asint(asfloat(asuint(asfloat(val))));
    
    val = test(val);

    outputBuffer[tid] = val + int(dot(vec2, vec4));
}