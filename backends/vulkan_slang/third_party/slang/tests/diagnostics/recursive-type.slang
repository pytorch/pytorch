//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-target spirv

// CHECK: error 41001:

struct Outer {
    Outer next; // non-pointer
    int y;
};
RWStructuredBuffer<Outer> Buf;

[numthreads(1,1,1)]
void csmain() {
     Buf[0].y = 0;
}