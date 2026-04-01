//TEST:SIMPLE(filecheck=CHECK): -target metal
//TEST:SIMPLE(filecheck=CHECK-ASM): -target metallib

// CHECK:     struct VOut{{.*}}
// CHECK-NEXT:{
// CHECK-NEXT:    float4 position{{.*}} {{\[\[}}position{{\]\]}};
// CHECK-NEXT:    float4 vertexColor{{.*}} {{\[\[}}user(_SLANG_ATTR){{\]\]}};
// CHECK-NEXT:    float2 vertexUV{{.*}} {{\[\[}}user(_SLANG_ATTR_1){{\]\]}};
// CHECK-NEXT:    float3 vertexNormal{{.*}} {{\[\[}}user(NORMAL){{\]\]}};
// CHECK-NEXT:};

// CHECK:     struct vertexInput{{.*}}
// CHECK-NEXT:{
// CHECK-NEXT:    float4 position{{.*}} {{\[\[}}attribute(0){{\]\]}};
// CHECK-NEXT:    float4 color{{.*}} {{\[\[}}attribute(1){{\]\]}};
// CHECK-NEXT:};

// CHECK: {{\[\[}}vertex{{\]\]}} VOut{{.*}} main_vertex(vertexInput{{.*}}{{\[\[}}stage_in{{\]\]}}, uint vid{{.*}}{{\[\[}}vertex_id{{\]\]}}, uint instanceID{{.*}} {{\[\[}}instance_id{{\]\]}})

// CHECK:     struct pixelOutput{{.*}}
// CHECK-NEXT:{
// CHECK-NEXT:    float4 output{{.*}}{{\[\[}}color(0){{\]\]}};
// CHECK-NEXT:};

// CHECK:     struct pixelInput{{.*}}
// CHECK-NEXT:{
// CHECK-NEXT:    float4 vertexColor{{.*}}  {{\[\[}}user(_SLANG_ATTR){{\]\]}};
// CHECK-NEXT:    float2 vertexUV{{.*}} {{\[\[}}user(_SLANG_ATTR_1){{\]\]}};
// CHECK-NEXT:    float3 vertexNormal{{.*}} {{\[\[}}user(NORMAL){{\]\]}};
// CHECK-NEXT:};

// CHECK: {{\[\[}}fragment{{\]\]}} pixelOutput{{.*}} main_fragment(pixelInput{{.*}} {{\[\[}}stage_in{{\]\]}}, float4 position{{.*}} {{\[\[}}position{{\]\]}})

// CHECK:     struct FragOut{{.*}}
// CHECK-NEXT:{
// CHECK-NEXT:    float4 color{{.*}}{{\[\[}}color(0){{\]\]}};
// CHECK-NEXT:    float depth{{.*}} {{\[\[}}depth(any){{\]\]}};
// CHECK-NEXT:};

// CHECK-ASM: define {{.*}} @main_vertex
// CHECK-ASM: define {{.*}} @main_fragment
// CHECK-ASM: define {{.*}} @main_fragment1

struct VIn
{
    float4 position : POSITION;
    float4 color : COLOR;
    uint vid : SV_VertexID;
    uint instanceID : SV_InstanceID;
}

struct VOut
{
    float4 position : SV_Position;
    float4 vertexColor;
    float2 vertexUV;
    float3 vertexNormal : NORMAL;
}

[shader("vertex")]
VOut main_vertex(VIn vertexIn)
{
    VOut vertexOut;
    vertexOut.position = vertexIn.position;
    vertexOut.vertexColor = vertexIn.color;
    vertexOut.vertexUV = float2(0.0, 1.0);
    return vertexOut;
}

[shader("fragment")]
float4 main_fragment(VOut fragmentIn) : SV_Target
{
    return fragmentIn.vertexColor;
}

struct FragOut
{
    float4 color : SV_Target;
    float depth : SV_Depth;
}

[shader("fragment")]
FragOut main_fragment1(VOut fragmentIn)
{
    FragOut fragOut;
    fragOut.color = fragmentIn.vertexColor;
    fragOut.depth = 0.5;
    return fragOut;
}