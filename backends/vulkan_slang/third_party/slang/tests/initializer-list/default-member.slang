//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

struct DefaultMember {
  int x = 0;
  int y = 1;
  // compiler synthesizes:
  // __init(int x = 0, int y = 1);
}

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
void test()
{
   DefaultMember m1 = {1}; // calls `__init(1)`, initialized to `{1,1}`.

   // BUFFER: 1
   outputBuffer[0] = m1.x;

   // BUFFER-NEXT: 1
   outputBuffer[1] = m1.y;

   DefaultMember m2 = {1,2}; // calls `__init(1,2)`, initialized to `{1,2}`.

   // BUFFER-NEXT: 1
   outputBuffer[2] = m2.x;
   // BUFFER-NEXT: 2
   outputBuffer[3] = m2.y;
}

[shader("compute")]
void computeMain()
{
    test();
}
