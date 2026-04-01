//TEST:SIMPLE(filecheck=CHECK): -target spirv

// ViewIndex builtin should be declared twice, once for vertex and once for fragment shader.
// Because the fragment shader builtin needs Flat, and vertex shader does not.

// CHECK: OpDecorate %{{.*}} BuiltIn ViewIndex
// CHECK: OpDecorate %{{.*}} BuiltIn ViewIndex

[shader("vertex")]
float4 vert(int _viewportIndex : SV_ViewID):SV_Position
{
    return float4(_viewportIndex);
}

[shader("fragment")]
float4 frag(int _viewportIndex : SV_ViewID) : SV_Target
{
    return float4(_viewportIndex, 0, 0, 1);
}