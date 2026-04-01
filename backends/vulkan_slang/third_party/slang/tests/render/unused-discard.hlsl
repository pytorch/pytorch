//DISABLED_TEST(smoke,render):COMPARE_HLSL_GLSL_RENDER:
//DISABLED_TEST(smoke,render):COMPARE_HLSL_GLSL_RENDER: -dx12

// This is a basic test case for cross-compilation behavior.
//
// We will define distinct HLSL and GLSL entry points,
// but the two will share a dependency on a file of
// pure Slang code that provides the actual shading logic.


#if defined(__HLSL__)

// Pull in Slang code depdendency using extended syntax:
__import unused_discard;

cbuffer Uniforms
{
	float4x4 modelViewProjection;
};

struct AssembledVertex
{
	float3	position;
	float3	color;
};

struct CoarseVertex
{
	float3	color;
};

struct Fragment
{
	float4 color;
};

// Vertex  Shader

struct VertexStageInput
{
	AssembledVertex assembledVertex	: A;
};

struct VertexStageOutput
{
	CoarseVertex	coarseVertex	: CoarseVertex;
	float4			sv_position		: SV_Position;
};

VertexStageOutput vertexMain(VertexStageInput input)
{
	VertexStageOutput output;

	float3 position = input.assembledVertex.position;
	float3 color	= input.assembledVertex.color;

	output.coarseVertex.color = color;
	output.sv_position = mul(modelViewProjection, float4(position, 1.0));

	return output;
	
}

// Fragment Shader

struct FragmentStageInput
{
	CoarseVertex	coarseVertex	: CoarseVertex;
};

struct FragmentStageOutput
{
	Fragment fragment	: SV_Target;
};

FragmentStageOutput fragmentMain(FragmentStageInput input)
{
	FragmentStageOutput output;

	float3 color = input.coarseVertex.color;

	color = transformColor(color);

	doConditionalDiscard(color);

	output.fragment.color = float4(color, 1.0);

	return output;
}

#elif defined(__GLSL__)

#version 420

float saturate(float x)
{
    return clamp(x, float(0), float(1));	
}

vec3 transformColor(vec3 color)
{
	vec3 result;

	result.x = sin(20.0 * (color.x + color.y));
	result.y = saturate(cos(color.z * 30.0));
	result.z = sin(color.x * color.y * color.z * 100.0);

	result = 0.5 * (result + 1);

	return result;
}

uniform Uniforms
{
	mat4x4 modelViewProjection;
};

#define ASSEMBLED_VERTEX(QUAL)		\
	/* */

#define V2F(QUAL)									\
	layout(location = 0) QUAL vec3 coarse_color;	\
	/* */

// Vertex  Shader

#ifdef __GLSL_VERTEX__

layout(location = 0)
in vec3 assembled_position;

layout(location = 1)
in vec3 assembled_color;

V2F(out)

void main()
{
	vec3 position = assembled_position;
	vec3 color	= assembled_color;

	coarse_color = color;
//	gl_Position = modelViewProjection * vec4(position, 1.0);
	gl_Position = vec4(position, 1.0) * modelViewProjection;
}

#endif

#ifdef __GLSL_FRAGMENT__

void doConditionalDiscard(vec3 color)
{
	if(color.x < 0.5)
		discard;
}

V2F(in)

layout(location = 0)
out vec4 fragment_color;

void main()
{
	vec3 color = coarse_color;

	color = transformColor(color);

	doConditionalDiscard(color);

	fragment_color = vec4(color, 1.0);
}


#endif

#endif
