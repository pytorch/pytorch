// unbounded-arrays.hlsl

//TEST:COMPARE_HLSL:-profile cs_5_1 -entry main
//TEST:REFLECTION:-profile cs_5_1 -target hlsl -no-codegen -D__SLANG__

//
// This test is trying to make sure that we correctly compute
// reflection/layout information for shaders that make use
// of unbounded arrays of resources.
//
// We will begin by declaring various "simple" global arrays
// of resource/sampler types and try out variations on binding
// them to registers/spaces or not.
//
// We will want to confirm that Slang generates the bindings
// we expect, and we will do this by enforcing explicit
// bindings on all parameters in the HLSL baseline we'll
// compare against:
//

    #ifdef __SLANG__
        #define REGISTER(x, y) /* empty */
    #else
        #define REGISTER(x,y) : register(x,y)
        #define aa aa_0
        #define b0 b0_0
        #define b1 b1_0
        #define bb bb_0
        #define c0 c0_0
        #define cc cc_0
        #define data data_0
    #endif

// First, let's just declare a simple unbounded array of samplers.
// We expect this to be given its own register and space (for D3D12)
//

    SamplerState aa[] REGISTER(s0, space2);

//
// Next, we will try to declare an array of resources with an explicit
// `register` binding. This should be set to start at that register in
// space zero (the default space), and should therefore "claim" all
// registers from that point on.
//

    Texture2D bb[] : register(t2);

//
// If we have assigned register t2 and beyond in space zero to `bb`,
// then we should still be able to put other resources in there explicitly:
//

    Texture2D b0 : register(t0);
    Texture2D b1 : register(t1, space0);

//
// It should also be possible to give an unbounded array an explicit
// register and space, and again it should be poossible to fill
// in the space before the unbounded array:
//

    TextureCube cc[] : register(t1, space1);
    Texture2D c0 : register(t0, space1);

//
// As a final detail, we should allow the user to specify the space
// and no register, which should be interpreted as requesting *any*
// register in the given space.
//
// TODO: Implement support for this case.
//

//    SamplerState dd[] : register(space5);

//
// With the simple cases out of the way, we will look at cases
// that involve structures and nested arrays.
//
// The first case we'll test is a structure type that contains
// two or more resources:
//

    struct X
    {
        Texture3D t;
        SamplerState s;
    };

//
// The simple case should Just Work, even though the same
// syntax will fail when used with fxc (so we have to
// provide a hand-written expansion for the baseline).
//

    #ifdef __SLANG__
        X ee[];
    #else
        Texture3D ee_t_0[] REGISTER(t0, space3);
        SamplerState ee_s_0[] REGISTER(s0, space4);
    #endif

//
// TODO: we should probably test interactions with explicit
// bindings for a structrure.
//
// TODO: we should also test cases that mix resource and
// non-resource types, but we can't currently have an unbounded
// array of uniform data (in HLSL at least).
//
// TODO: should test arrays-of-arrays cases.
//


//
// We'll close things out with a dummy entry point just
// to allow this file to be compiled with fxc/dxc.
//

    float4 use(Texture2D t, SamplerState s, float4 u)
    {
        return t.SampleLevel(s, u.xy + u.z, u.w);
    }

    float4 use(Texture3D t, SamplerState s, float4 u)
    {
        return t.SampleLevel(s, u.xyz, u.w);
    }

    float4 use(TextureCube t, SamplerState s, float4 u)
    {
        return t.SampleLevel(s, u.xyz, u.w);
    }

    RWStructuredBuffer<float4> data;

    [numthreads(4,1,1)]
    void main(uint3 tid : SV_DispatchThreadID)
    {
        int idx = int(tid.x);
        float4 tmp = data[idx];

        SamplerState s = aa[idx];

        tmp = use(bb[idx],  s, tmp);
        tmp = use(b0,       s, tmp);
        tmp = use(b1,       s, tmp);
        tmp = use(cc[idx],  s, tmp);
        tmp = use(c0,       s, tmp);

#ifdef __SLANG__
        tmp = use(ee[idx].t, ee[idx].s, tmp);
#else
        tmp = use(ee_t_0[idx], ee_s_0[idx], tmp);
#endif
        data[idx] = tmp;
    }
