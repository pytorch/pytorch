//TEST:REFLECTION:-profile sm_5_1 -stage fragment -target hlsl -no-codegen

// This test is to confirm that we do not allocate a "default"
// space/set for global shader parameters unless it is
// really required. In particular, if there are global-scope
// resource parameters *but* they are all explicitly bound,
// then a default space isn't needed.


// An explicitly-bound global texture.
Texture2D a : register(t0, space99);

// An implicitly-bound global parameter block.
//
// This parameter should be given `space0`, because
// it is the first available space after all explicitly-bound
// parameters have claimed their registers/spaces.
//
struct B { Texture2D b; }
ParameterBlock<B> b;

float4 main() : SV_Target
{ return 0.0; }
