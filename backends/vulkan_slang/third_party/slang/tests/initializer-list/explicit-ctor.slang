//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct ExplicitCtor
{
  int x;
  int y;
  __init(int x)
  {
    this.x = x;
    this.y = x + 5;
  }
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
void test()
{
  // case 1: initialized with synthesized ctor call using legacy logic to form arguments,
  // and `c1` is now `{0,0}`.
  ExplicitCtor c1 = {4};

  // BUFFER: 4
  outputBuffer[0] = c1.x;
  // BUFFER-NEXT: 9
  outputBuffer[1] = c1.y;


  ExplicitCtor c2 = ExplicitCtor(10);

  // BUFFER: A
  outputBuffer[2] = c2.x;
  // BUFFER-NEXT: F
  outputBuffer[3] = c2.y;
}

[shader("compute")]
void computeMain()
{
    test();
}
