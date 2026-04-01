//TEST(smoke,compute):COMPARE_COMPUTE(filecheck-buffer=CHECK):-cpu

// CHECK:      0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<int> outputBuffer;

groupshared int myInt;

void set(__ref groupshared int i){ i = 1; }
int use(__ref groupshared int i){ return i; }

[numthreads(4, 1, 1)]
void computeMain(uint i : SV_GroupIndex)
{
    set(myInt);
    use(myInt);
    outputBuffer[i] = i;
}
