//TEST:SIMPLE(filecheck=CHECK): -target spirv -entry main -stage fragment

// CHECK: OpCapability Geometry
// CHECK:  BuiltIn PrimitiveId

#version 450

out vec4 color;

void main()
{
    color = float4(gl_PrimitiveID, 0, 0, 1);
}