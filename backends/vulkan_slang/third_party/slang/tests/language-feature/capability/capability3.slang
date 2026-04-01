//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly -entry main -stage compute
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target spirv -emit-spirv-directly -entry main -stage compute -ignore-capabilities -skip-spirv-validation
// CHECK_IGNORE_CAPS-NOT: error 36108

// Test that capabilities can be declared on module.
[require(glsl)]
[require(spirv)]
module test;

RWStructuredBuffer<int> sideEffect;

// CHECK: error {{(36107|36108)}}
[require(glsl, _sm_4_0)]
public void use1()
{
    __target_switch
    {
    case glsl:
        return;
    }
}

void use2Sub()
{
    __target_switch
    {
    case glsl:
        sideEffect[1] = 1;
    }
}
// CHECK: error {{(36107|36108)}}
[require(spirv, spirv_1_0)]
public void use2()
{
    use2Sub();
}

void main()
{
    use1();
    use2();
}
