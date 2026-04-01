//TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry main -profile cs_6_0 -no-hlsl-pack-constant-buffer-elements
//TEST:SIMPLE(filecheck=DXIL): -target dxil -entry main -profile cs_6_0 -no-hlsl-pack-constant-buffer-elements

// DXIL: define void @main()

// Test that -no-hlsl-pack-constant-buffer-elements prevents packing of elements in the output HLSL code.

// CHECK-DAG: cbuffer {{.*}} :
cbuffer MyCB
{
    float member0;
    float member1;
}

RWStructuredBuffer<float> MyBuffer;

// CHECK-DAG: {{.*}} = member0{{.*}} + member1{{.*}};
[numthreads(1,1,1)]
void main()
{
    MyBuffer[0] = member0 + member1;
}
