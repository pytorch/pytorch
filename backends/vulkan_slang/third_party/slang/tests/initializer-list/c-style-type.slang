//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct CLike
{
  int x;
  int y;
  // compiler synthesizes:
  // __init(int x, int y);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
void test()
{
  // case 1: initialized with synthesized ctor call using legacy logic to form arguments,
  // and `c1` is now `{0,0}`.
  CLike c1 = {};

  // case 2: initialized with legacy initializaer list logic, `c1` is now `{1,0}`:
  CLike c2 = {1};

  // case 3: initilaized with ctor call `CLike(1,2)`, `c3` is now `{1,2}`:
  CLike c3 = {1, 2};

  // BUFFER: 0
  outputBuffer[0] = c1.x;
  // BUFFER-NEXT: 0
  outputBuffer[1] = c1.y;
  // BUFFER-NEXT: 1
  outputBuffer[2] = c2.x;
  // BUFFER-NEXT: 0
  outputBuffer[3] = c2.y;
  // BUFFER-NEXT: 1
  outputBuffer[4] = c3.x;
  // BUFFER-NEXT: 2
  outputBuffer[5] = c3.y;
}

[shader("compute")]
void computeMain()
{
    test();
}
