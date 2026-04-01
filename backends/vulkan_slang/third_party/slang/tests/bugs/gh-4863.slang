//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-compute -output-using-type
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-vk -compute -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

interface IFoo
{
  property float bar1 {get; set;}
};

struct Foo : IFoo
{
  uint4 data = 0;
  property float bar1
  {
    get { return asfloat(data.w); }
    set { data.w = asuint(newValue); }
  }
};

void doThing(inout Foo foo, uint2 ipos)
{
  foo.bar1 = 2;
  foo.bar1 += 1.0f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint2 ipos : SV_DispatchThreadID)
{
  Foo foo = {};
  doThing(foo, ipos);
  // BUFFER: 3
  outputBuffer[0] = (int)foo.bar1;
}
