// sv-coverage.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment -emit-spirv-via-glsl

float4 main(
	in float4 color : COLOR,
	in uint inputCoverage : SV_Coverage,
	out uint outputCoverage : SV_Coverage)
	: SV_Target
{
    // CHECK: OpDecorate %gl_SampleMask{{.*}} BuiltIn SampleMask
    // CHECK: %gl_SampleMask{{.*}} = OpVariable %_ptr_Input__arr_int_{{u?}}int_1 Input
	outputCoverage = inputCoverage ^ 1;
	return color;
}
