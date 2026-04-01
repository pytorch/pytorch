//TEST(compute):COMPARE_COMPUTE: -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4),name=buf
RWStructuredBuffer<float4> buf;

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(1,1,1)]
void computeMain()
{
    float sum = 0.0f;
    
    buf[0].wxyz = float4(1.0, 0.0, 0.0, 0.0);
    sum += buf[0].w;
    
    buf[0].xwyz = float4(0.0, 1.0, 0.0, 0.0);
    sum += buf[0].w;

    buf[0].xywz = float4(0.0, 0.0, 1.0, 0.0);
    sum += buf[0].w;

    buf[0].xyzw = float4(0.0, 0.0, 0.0, 1.0);
    sum += buf[0].w;

    outputBuffer[0] = int(sum);
}
