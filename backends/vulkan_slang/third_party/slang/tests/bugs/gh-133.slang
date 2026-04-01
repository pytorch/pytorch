//TEST_DISABLED:CROSS_COMPILE: -profile ps_5_0 -entry main -target spirv-assembly

// Ensure that an integer output from
// a fragment shader doesn't get a `flat` qualifier

struct Fragment
{
	uint foo;
};

cbuffer U
{
	uint bar;
}

Fragment main() : SV_Target
{
	Fragment result;
	result.foo = bar;
	return result;
}
