//TEST:SIMPLE(filecheck=CHECK):-target glsl -profile ps_4_0 -entry main -fvk-t-shift 10 all  -fvk-s-shift 100 all -fvk-u-shift 100 all -fvk-b-shift 1000 all

// CHECK:layout(binding = 10)
// CHECK-NEXT:uniform texture2D texture0_0;

// CHECK:layout(binding = 100)
// CHECK-NEXT:uniform sampler sampler0_0;

// CHECK:layout(binding = 11, set = 2)
// CHECK-NEXT:uniform texture2D texture1_0;

// CHECK:layout(binding = 101, set = 2)
// CHECK-NEXT:uniform sampler sampler1_0;

// CHECK: layout(push_constant)
// CHECK-NEXT: layout(std430) uniform

// CHECK:layout(binding = 1004)
// CHECK-NEXT:layout(std140) uniform

// CHECK:layout(binding = 1003)
// CHECK-NEXT:layout(std140) uniform

// CHECK:layout(binding = 1002)
// CHECK-NEXT:layout(std140) uniform

// CHECK:layout(binding = 1001)
// CHECK-NEXT:layout(std140) uniform

// CHECK: struct GlobalParams
// CHECK-NEXT: {
// CHECK-NEXT: float g_value
// CHECK-NEXT: }

// CHECK:layout(binding = 1000)
// CHECK-NEXT:layout(std140) uniform

Texture2D texture0; 
SamplerState sampler0;

Texture2D texture1 : register(t1, space2); 
SamplerState sampler1 : register(s1, space2); 

float g_value;

cbuffer ConstantBufferA 
{ 
    float constantA;
}; 

cbuffer ConstantBufferB 
{ 
    float constantB;
}; 

cbuffer ConstantBufferC
{
    float constantC; 
}; 

cbuffer ConstantBufferD
{ 
    float constantD;
}; 

struct StructA 
{ 
    float a;
};

[[ vk::push_constant ]]
StructA pushConstantA;

struct PixelInput
{ 
    float4 t : TEXCOORD0; 
}; 

struct PixelOutput
{ 
    float4 color : SV_Target0; 
}; 

float4 use(Texture2D t, SamplerState s) { return t.SampleLevel(s, 0.0, 0.0); }

PixelOutput main(PixelInput i)
{ 
    PixelOutput o ; 
    float4 b = use(texture0, sampler0) + use(texture1, sampler1);
    float a = pushConstantA.a + constantD + constantC + constantB + constantA + g_value + length(b);
    o.color = float4(1, 2, 3, 4) * a;
    return o ; 
} 

