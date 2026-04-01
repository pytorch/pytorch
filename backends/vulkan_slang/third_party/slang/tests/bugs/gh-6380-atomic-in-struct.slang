//TEST:SIMPLE(filecheck=CHECK): -target spirv
//CHECK: OpEntryPoint

struct Test {
    Atomic<int> test;
}

RWStructuredBuffer<Test> data;

[shader("compute")]
[numthreads(1,1,1)]
public void main() {
    Test t = Test(1);
    data[0].test.increment();
}