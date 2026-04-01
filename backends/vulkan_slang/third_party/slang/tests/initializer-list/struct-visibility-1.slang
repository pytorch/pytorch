
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj -vk
//TEST(compute):COMPARE_COMPUTE(filecheck-buffer=BUFFER):-shaderobj

public struct Visibility
{
  internal int x = 1;
  public int y = 5;

  int getX() { return x; }
}


//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;
void test(inout uint index)
{
  Visibility t1 = {}; // OK, initialized to {1,5} via ctor call.
  // BUFFER: 1
  outputBuffer[index++] = t1.getX();
  // BUFFER-NEXT: 5
  outputBuffer[index++] = t1.y;

  Visibility t2 = {1}; // OK, initialized to {1,1} via ctor call.
  // BUFFER-NEXT: 1
  outputBuffer[index++] = t2.getX();
  // BUFFER-NEXT: 1
  outputBuffer[index++] = t2.y;
}

internal struct Visibility2
{
  // Visibility3 type is considered as C-style struct.
  // Because all members have the same visibility as the type.
  // Therefore we will attempt the legacy fallback logic for
  // initializer-list syntax.
  // Note that c-style structs can still have init exprs on members.
  internal int x;
  internal int y = 2;
  // compiler synthesizes:
  // internal __init(int x, int y = 2);
}

internal void test2(inout uint index)
{
  Visibility2 t = {3, 4}; // OK, initialized to {3,4} via ctor call.
  // BUFFER-NEXT: 3
  outputBuffer[index++] = t.x;
  // BUFFER-NEXT: 4
  outputBuffer[index++] = t.y;

  Visibility2 t1 = {1}; // OK, initialized to {1,2} via ctor call.
  // BUFFER-NEXT: 1
  outputBuffer[index++] = t1.x;
  // BUFFER-NEXT: 2
  outputBuffer[index++] = t1.y;

  Visibility2 t2 = {}; // OK, initialized to {0, 2} via legacy logic.
  // BUFFER-NEXT: 0
  outputBuffer[index++] = t2.x;
  // BUFFER-NEXT: 2
  outputBuffer[index++] = t2.y;
}

internal struct Visibility3
{
  // Visibility4 type is considered as C-style struct.
  // And we still synthesize a ctor for member initialization.
  // Because Visibility4 has no public members, the synthesized
  // ctor will take 0 arguments.
  internal int x = 1;
  internal int y = 2;
  // compiler synthesizes:
  // internal __init(int x = 1, int y = 2);
}

internal void test3(inout uint index)
{
  Visibility3 t = {0, 0}; // OK, initialized to {0,0} via ctor call.
  // BUFFER-NEXT: 0
  outputBuffer[index++] = t.x;
  // BUFFER-NEXT: 0
  outputBuffer[index++] = t.y;

  Visibility3 t1 = {3}; // OK, initialized to {3,2} via ctor call.
  // BUFFER-NEXT: 3
  outputBuffer[index++] = t1.x;
  // BUFFER-NEXT: 2
  outputBuffer[index++] = t1.y;

  Visibility3 t2 = {}; // OK, initialized to {1,2} via ctor call.
  // BUFFER-NEXT: 1
  outputBuffer[index++] = t2.x;
  // BUFFER-NEXT: 2
  outputBuffer[index++] = t2.y;
}

[shader("compute")]
void computeMain()
{
    uint index = 0;
    test(index);
    test2(index);
    test3(index);
}
