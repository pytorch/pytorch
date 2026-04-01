// parameter-block-explicit-space.slang

//TEST:REFLECTION:-D__SLANG__ -stage fragment -entry main -profile sm_5_1 -target hlsl -no-codegen
//TEST:COMPARE_HLSL:-stage fragment -entry main -profile sm_5_1

#ifdef __SLANG__
struct A
{
    float4          au;
    Texture2D       at1;
    Texture2D       at2;
    SamplerState    as;
}

struct X
{
    float4 xu;    
}

struct B
{
    float4              bu;
    Texture2D           bt;
    SamplerState        bs;

    // TODO: This line leads to a crash
//    ConstantBuffer<X>   bx;
}



[[vk::binding(0,2)]]
ParameterBlock<A> a : register(space2);

[[vk::binding(0,3)]]
ParameterBlock<B> b : register(space3);

float4 use(float4 val) { return val; }
float4 use(Texture2D t, SamplerState s)
{
    return t.Sample(s, 0.0);
}

float4 main() : SV_Target
{
    return use(a.au)
        + use(a.at1, a.as)
        + use(a.at2, a.as)
        + use(b.bu)
        + use(b.bt, b.bs)
        // + use(b.bx.xu)
        ;
}

#else

#define A A_0
#define a a_0
#define au au_0
#define at1 a_at1_0
#define at2 a_at2_0
#define as a_as_0

#define B B_0
#define b b_0
#define bu bu_0
#define bt b_bt_0
#define bs b_bs_0

struct A
{
    float4 au;
};
cbuffer a            : register(b0, space2)
{ A a; }
Texture2D       at1  : register(t0, space2);
Texture2D       at2  : register(t1, space2);
SamplerState    as  : register(s0, space2);

struct B
{
    float4      bu;
};
cbuffer b           : register(b0, space3)
{ B b; }
Texture2D       bt  : register(t0, space3);
SamplerState    bs  : register(s0, space3);

float4 use(float4 val) { return val; }
float4 use(Texture2D t, SamplerState s)
{
    return t.Sample(s, 0.0);
}

float4 main() : SV_TARGET
{
    return use(a.au)
        + use(at1, as)
        + use(at2, as)
        + use(b.bu)
        + use(bt, bs);
}

#endif
