// Test if generic arguments using arithmetics are folded properly.

//TEST:SIMPLE(filecheck=HLSL): -stage compute -entry computeMain -target hlsl

RWStructuredBuffer<float> outputBuffer;

__generic<let ArraySize:int>
struct MyStruct
{
  int elems[ArraySize];
};

__generic<let ArraySize:int>
MyStruct<ArraySize / 2> Reduce(MyStruct<ArraySize> o)
{
  MyStruct<ArraySize / 2> result;
  //HLSL:[[ReturnType:MyStruct_[0-9]*]] Reduce
  //HLSL:[[ReturnType]] result

  result.elems[0] = o.elems[0];

  // Error happened here as the return type differs
  // from the type of "result".
  return result;
}

[numthreads(1, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
  MyStruct<4> a;
  a.elems[0] = 2;

  let result = Reduce(a);
  outputBuffer[0] = result.elems[0];
}
