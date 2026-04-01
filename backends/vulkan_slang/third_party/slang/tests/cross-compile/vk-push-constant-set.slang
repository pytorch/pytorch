// vk-push-constant.slang

// Test to confirm that a `[[vk::push_constant]]` buffer
// doesn't end up reserving `space=0` for global scope
// parameters and shifting a parameer block over to
// `space=1`.

//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -stage fragment

struct S
{
	float4 v;
}

[[vk::push_constant]]
ConstantBuffer<S> x;

// CHECK: OpDecorate %y Binding 0
// CHECK: OpDecorate %y DescriptorSet 0

ParameterBlock<S> y;

float4 main() : SV_Target
{
    return x.v + y.v;
}
