//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct PartialInit {
  // warning: not all members are initialized.
  // members should either be all-uninitialized or all-initialized with
  // default expr.
  int x;
  int y = 1;
  // compiler synthesizes:
  // __init(int x, int y = 1);
}

struct PartialInit2 {
  int x = 1;
  int y;
  // __init(int x, int y);
}
//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
void test()
{
  PartialInit p1 = {2}; // calls `__init`, result is `{2,1}`.
  // BUFFER: 2
  outputBuffer[0] = p1.x;
  // BUFFER-NEXT: 1
  outputBuffer[1] = p1.y;


  PartialInit p2 = {2, 3}; // calls `__init`, result is {2, 3}
  // BUFFER-NEXT: 2
  outputBuffer[2] = p2.x;
  // BUFFER-NEXT: 3
  outputBuffer[3] = p2.y;

  PartialInit2 p3 = {4, 5}; // calls `__init`, result is {4, 5}
  // BUFFER-NEXT: 4
  outputBuffer[4] = p3.x;
  // BUFFER-NEXT: 5
  outputBuffer[5] = p3.y;
}

[shader("compute")]
void computeMain()
{
    test();
}
