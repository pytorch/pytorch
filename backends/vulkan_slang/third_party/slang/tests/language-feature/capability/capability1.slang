//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry main2 -stage compute
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry main2 -stage compute -ignore-capabilities
// CHECK_IGNORE_CAPS-NOT: error 36104

[require(spvShaderClockKHR)]
void leafFunc1() {}

[require(spvShaderNonUniform)]
void leafFunc2() {}

void caller()
{
    leafFunc1();
    leafFunc2();
}

[require(spirv, shaderclock)]
// CHECK: error 36104:
void main1()
{
    caller(); // Error, shaderclock does not imply spvShaderNonUniform.
}


[require(spirv, shaderclock)]
void main2()
{
    leafFunc1(); // OK, shaderclock implies spvShaderClockKHR.
}
