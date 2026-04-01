//TEST:SIMPLE(filecheck=CHECK): -target spirv -g3 -emit-spirv-directly -profile sm_6_6+spirv_1_5
// CHECK:  OpString "-target spirv {{.*}} -capability spirv_1_5
[numthreads(1,1,1)]
void computeMain()
{}