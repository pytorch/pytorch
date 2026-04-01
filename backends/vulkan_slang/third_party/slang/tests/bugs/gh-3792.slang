//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUF): -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

interface From<X> {
  __init(const X x);
}
extension float: From<float> {
  __init(const float x) { this = x; }
}
T test<T : From<float>>(float v)
{
    T u;
    u = T(5.0f);
    return u;
}

RWStructuredBuffer<float> outputBuffer;
[numthreads(1,1,1)]
void computeMain()
{
    float v = test<float>(5.0);
    //BUF: 5.0
    outputBuffer[0] = v;
}