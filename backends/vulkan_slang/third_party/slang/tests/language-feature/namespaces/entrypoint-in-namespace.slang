//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry Pixel.MyType.Main -stage fragment

// Test that we can compile an entrypoint defined in a namespace.

// CHECK: OpEntryPoint
struct PSInput
{
	float4 color : COLOR;
};

namespace Pixel
{
    struct MyType
    {
        static float4 Main(PSInput input) : SV_TARGET
        {
            return input.color;
        }
    }    
}