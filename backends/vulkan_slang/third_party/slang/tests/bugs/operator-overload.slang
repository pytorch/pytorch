//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

// Tests operator overloading works in user space.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

struct Vec2d
{   
    float x, y;
};

Vec2d operator+(Vec2d a, Vec2d b)
{
    return {a.x + b.x, a.y + b.y};
}

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;
    
    Vec2d a = { 1, -index + 3 };
    Vec2d b = { index - 4, index * index };
    
    Vec2d c = a + b;
    
    int r = int(c.x + c.y);
    
    outputBuffer[dispatchThreadID.x] = int(r);
}
