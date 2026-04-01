//TEST_DISABLED:CROSS_COMPILE: -profile ps_5_0 -entry main -target spirv-assembly

// Note: disabled, pending adding an IR-based definition for `GetDimensions`.

// Ensure that `GetDimensions` with `mipCount` output works
// on a `Texture2D`

layout(binding = 1)
Texture2D t;

float4 main() : SV_Target
{
	uint x, y, mipCount;
    t.GetDimensions(0, x, y, mipCount);

    return float4(x, y, mipCount, 0);
}
