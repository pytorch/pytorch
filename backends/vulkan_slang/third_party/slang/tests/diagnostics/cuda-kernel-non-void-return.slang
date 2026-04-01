//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):

// Simple check to see if the compiler throws an error if a CUDA kernel is declared with non-void return type.

[CudaKernel]
float myBadKernel(TensorView<float> t1, TensorView<float> t2)
{
    // CHECK:      tests/diagnostics/cuda-kernel-non-void-return.slang([[@LINE-2]]): error 31213: return type of a CUDA kernel function cannot be non-void.
    // CHECK-NEXT:     float myBadKernel(TensorView<float> t1, TensorView<float> t2)
    // CHECK-NEXT:           ^~~~~~~~~~~
}

[CudaKernel]
void myGoodKernel(TensorView<float> t1, TensorView<float> t2)
{

}