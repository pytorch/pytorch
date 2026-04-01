//TEST:SIMPLE(filecheck=CHECK):-target spirv -emit-spirv-directly -entry main -stage fragment
struct PerPrimitive {
    int primitive_id : SV_PrimitiveID;
    int layer : SV_RenderTargetArrayIndex;
};

// CHECK: OpDecorate %gl_PrimitiveID Flat
// CHECK: OpDecorate %gl_Layer Flat
[shader("fragment")]
uint main(PerPrimitive data) : SV_Target {
    return data.primitive_id + data.layer;
}