//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry computeMain -stage compute -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: error 36111



[require(spirv, hlsl)]
// CHECK: error 36111
// CHECK_IGNORE_CAPS-NOT: error 36111
void computeMain()
{

}
