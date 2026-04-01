//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: ([[# @LINE+1]]): error 38029
Texture2D<float4x4> t1;

// CHECK: ([[# @LINE+1]]): error 38029
RWBuffer<Sampler1D> t2;