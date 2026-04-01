// struct-default-init.slang
//TEST:SIMPLE(filecheck=HLSL): -target hlsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=GLSL): -target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=METAL): -target metal -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CPP): -target cpp -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=SPIRV): -target spirv -entry computeMain -stage compute

// HLSL: computeMain
// GLSL: main
// METAL: computeMain
// CPP: computeMain
// SPIRV: OpEntryPoint

struct PowActivationEx : IDifferentiable
{
	float power;
}
RWStructuredBuffer<float> return_value;

[numthreads(256, 1, 1)]
void computeMain()
{
    PowActivationEx args;
	return_value[0] = args.power;
}