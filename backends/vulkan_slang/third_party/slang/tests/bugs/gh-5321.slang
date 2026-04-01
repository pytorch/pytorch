//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: error {{.*}} compile-time constant

uniform uint FillType_Solid = 0u;

[shader("compute")]
float cs(uniform uint x)
    : SV_Target
{
    switch (x)
    {
    case FillType_Solid:
        return 0.0;
    }
    return 0.0;
}