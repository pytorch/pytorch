//TEST:SIMPLE(filecheck=CHECK):-target spirv-assembly -entry main -profile ps_5_0

// CHECK: OpDecorate %pblock Binding 0
// CHECK: OpDecorate %pblock DescriptorSet 0
struct E
{
};

struct P
{
	E em;
	float4 param;
};
ParameterBlock<P> pblock;

float4 main(float4 pos : SV_POSITION)
{
	return pblock.param;
}