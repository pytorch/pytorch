//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHK):-vk -compute -shaderobj -output-using-type

// This tests if the scope(::) is recognized for a local variable declaration.

//TEST_INPUT: ubuffer(data=[0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[UnscopedEnum]
enum Number {
  First = 1,
  Second,
};

static ::Number foo = First;

[numthreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID: SV_DispatchThreadID)
{
  // The scope(::) should be recognized
  static ::Number bar = Second;

  //CHK:1
  //CHK-NEXT:2
  outputBuffer[0] = int(foo);
  outputBuffer[1] = int(bar);
}

