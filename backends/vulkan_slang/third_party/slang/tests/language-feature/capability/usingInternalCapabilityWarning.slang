//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry main -stage compute -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: warning 36113
// CHECK: .slang(5):{{.*}}warning 36113: {{.*}}_GL_NV_ray_tracing_motion_blur{{.*}}GL_NV_ray_tracing_motion_blur
[require(_GL_NV_ray_tracing_motion_blur)]
[numthreads(1,1,1)]
void computeMain()
{
}