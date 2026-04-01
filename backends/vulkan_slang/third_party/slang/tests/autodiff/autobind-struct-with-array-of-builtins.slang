//TEST:SIMPLE(filecheck=CUDA): -target cuda -line-directive-mode none
//TEST:SIMPLE(filecheck=TORCH): -target torch -line-directive-mode none


[AutoPyBindCUDA]
[CUDAKernel]
void plain_copy(float3[4] input, TensorView<float> output)
{
    // CUDA: __global__ void __kernel__plain_copy(FixedArray<_VectorStorage_float3_0, 4>  input_0, TensorView output_0)
    // TORCH:    void __kernel__plain_copy(FixedArray<_VectorStorage_float3_0, 4>  _0, TensorView _1);

    // Get the 'global' index of this thread.
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();

    // If the thread index is beyond the input size, exit early.
    if (dispatchIdx.x >= 1)
        return;

    output[0] = input[0].x;
    output[1] = input[2].y;
    output[2] = input[3].z;
}
