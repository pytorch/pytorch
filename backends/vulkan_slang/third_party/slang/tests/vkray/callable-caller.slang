// callable-caller.slang

//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460 -capability GL_NV_ray_tracing -stage raygeneration -entry main -target spirv-assembly
//TEST:SIMPLE(filecheck=CHECK): -profile glsl_460 -capability GL_NV_ray_tracing -stage raygeneration -entry main -target spirv-assembly -emit-spirv-directly


import callable_shared;

cbuffer C
{
	uint shaderIndex;
};

RWTexture2D<float4> gImage;

void main()
{
	MaterialPayload payload;
	payload.albedo = 0;
	payload.uv = float2(DispatchRaysIndex().xy) / float2(DispatchRaysDimensions().xy);
	// CHECK: OpExecuteCallable
	CallShader(shaderIndex, payload);

	gImage[DispatchRaysIndex().xy] = payload.albedo;
}
