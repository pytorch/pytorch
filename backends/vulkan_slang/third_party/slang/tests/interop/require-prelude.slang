//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK): -target wgsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK): -target metal

// CHECK: #[compute]

[numthreads(1,1,1)]
void computeMain()
{
    __requirePrelude(R"(#[compute])");
}