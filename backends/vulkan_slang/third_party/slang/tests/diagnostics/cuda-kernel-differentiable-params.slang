//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Simple check to see if the compiler throws an error if a CUDA kernel is declared with non-void return type.

[CudaKernel]
[Differentiable]
void myBadKernel(float x, TensorView<float> t1, TensorView<float> t2)
{
    // CHECK:      tests/diagnostics/cuda-kernel-differentiable-params.slang([[@LINE-2]]): error 31214: differentiable kernel entry point cannot have differentiable parameters. Consider using DiffTensorView to pass differentiable data, or marking this parameter with 'no_diff'
    // CHECK-NEXT:     void myBadKernel(float x, TensorView<float> t1, TensorView<float> t2)
    // CHECK-NEXT:                            ^
}

[CudaKernel]
void myGoodKernel(float x, TensorView<float> t1, TensorView<float> t2)
{

}