//TEST_DISABLED:COMPARE_HLSL: -profile vs_5_0

// Check output for `do` loops

cbuffer C : register(b0)
{
	int n;
};

float4 main() : SV_Position
{
	float4 x = 0;
	int i = n;
	{
		do
		{
			x = (x + (float)1) * x;

			// Note(tfoley): The "right" thing here would be
			// `i--`, but that leads to a subtle difference
			// in the final code between just invokeing `fxc`
			// and invoking it on the Slang-generated output
			// (despite the generated HLSL for this line being
			// identical, modulo some `#line` directives).
			//
			// I'm using a binary operator that will yield
			// the same code with its operands swapped, just
			// to work around it. A better long-term fix
			// is to have this test be an end-to-end test
			// that we execute.
			i = i*i;
		} while((bool) i);
	}
    return x;
}
