//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=METALLIB): -target metallib

// Test that we can specailize mutable parameters based on the argument
// address space when generating code for Metal.

//TEST_INPUT: ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

// CHECK: void Test_out_param{{.*}}(int thread* value{{.*}})
// CHECK: void Test_out_param{{.*}}(int device* value{{.*}})
// CHECK: void Test_out_param{{.*}}(int threadgroup* value{{.*}})

// CHECK: void Test_out_param_wrapper{{.*}}(int thread* value{{.*}})
// CHECK: void Test_out_param_wrapper{{.*}}(int device* value{{.*}})
// CHECK: void Test_out_param_wrapper{{.*}}(int threadgroup* value{{.*}})

void Test_out_param(out int value)
{
    value = 1;
}

void Test_out_param_wrapper(out int value)
{
    Test_out_param(value);
}

[numthreads(1,1,1)]
void computeMain()
{
    // METALLIB: define void @computeMain

    int value = 0;
    Test_out_param_wrapper(value);
    outputBuffer[0] = value;

    Test_out_param_wrapper(outputBuffer[1]);

    static groupshared int sharedValue[2];
    Test_out_param_wrapper(sharedValue[0]);
    outputBuffer[2] = sharedValue[0];

    outputBuffer[0]++;
}