//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none

struct MyStruct
{
    float3x3 data;
    float3 vec;
};

struct MyStruct2
{
    float data;
};

[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy(MyStruct input, MyStruct2 input2, TensorView<float> output)
{
    // CUDA: __global__ void __kernel__plain_copy(U_StructStorage_MyStruct_0 input_0, MyStruct2_0 input2_0, TensorView output_0)
    // TORCH:    void __kernel__plain_copy(U_StructStorage_MyStruct_0 _0, MyStruct2_0 _1, TensorView _2);

    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input.data[0][0];
    output[1] = input.vec[1];
    output[2] = input.data[2][2];
}
