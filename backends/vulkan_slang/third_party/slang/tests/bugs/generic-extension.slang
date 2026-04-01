//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
  static This myAdd(This v, float val);   
}

__generic<let N : int>
extension vector<float, N> : IFoo
{
  static vector<float, N> myAdd(vector<float, N> v, float val)
  {
    return v + vector<float, N>(val);
  }
}

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
  int index = int(dispatchThreadID.x);
  float2 v = 1.0;
  v = float2.myAdd(v, 2.0);
  outputBuffer[index] = int(v.x);
}

