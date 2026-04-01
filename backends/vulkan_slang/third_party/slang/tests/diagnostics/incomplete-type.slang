//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

extern struct ExtType;

struct MyType
{
    ExtType arr[2];
}

// CHECK: ([[#@LINE+1]]): error 31204
ConstantBuffer<MyType> buffer;
