// single-shader-record.slang
//DIAGNOSTIC_TEST:SIMPLE: -profile sm_6_3 -stage closesthit -entry main -target spirv-assembly

struct ReflectionRay
{
    float4 color;
};

StructuredBuffer<float4> colors;

// There are two constant buffers with shader_record attibuting so should produce and error

layout(shaderRecordNV)
cbuffer ShaderRecord2
{
	uint shaderRecordID2;
}

[[vk::shader_record]]
cbuffer ShaderRecord
{
	uint shaderRecordID;
}

void main(
	BuiltInTriangleIntersectionAttributes 	attributes,
	in out ReflectionRay 					ioPayload)
{
	uint materialID = InstanceIndex()
		+ InstanceID()
		+ PrimitiveIndex()
		+ HitKind()
		+ shaderRecordID 
        + shaderRecordID2;

	float4 color = colors[materialID];

	color *= RayTCurrent() - RayTMin();

	ioPayload.color = color;
}
