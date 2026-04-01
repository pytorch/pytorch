//TEST_DISABLED:SIMPLE:

// Note: This test is disabled because we don't currently
// have support for generating code from files that use
// interfaces as parameter types.
//
// TODO: We need to add a check for this and generate an
// error!

// Confirm that basic `interface` syntax stuff type-checks.

// The example here is adapted from examples in Matt Pharr's
// chapter in GPU Gems: "An Introduction to Shader Interaces"

struct LightSample
{
	float3 C; // radiance
	float3 L; // direction	
};

interface Light
{
	LightSample illuminate(float3 P_world);
}

struct PointLight : Light
{
	float3 Plight_world;
	float3 C;

	LightSample illuminate(float3 P_world)
	{
		float3 delta = Plight_world - P_world;
		float3 L = normalize(delta);
		float distance = length(delta);

		LightSample result;
		result.L = L;
		result.C = C * (1 / (distance*distance));
		return result;
	}	
};

// using the concrete type directly
float3 A( float3 P_world, PointLight light )
{
	return light.illuminate(P_world).L;
}

// using the abstract interface type
float3 B( float3 P_world, Light light )
{
	return light.illuminate(P_world).L;
}

//
float3 Test(float3 P_world, PointLight pointLight, Light light)
{
	// dconcrete type expected, concrete type provided
	float3 a = A(P_world, pointLight);

	// abstract type expected, abstract type provided
	float3 b = B(P_world, light);

	// abstract type expected, concrete type provided
	float3 c = B(P_world, pointLight);

	// The remaining case (passing `Light` where `PointLight` is expected)
	// should be an error, so we want a distinct test for that.

	return a + b + c;
}