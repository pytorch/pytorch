//TEST:SIMPLE(filecheck=SPIRV): -entry main -stage compute -target spirv
//TEST:SIMPLE(filecheck=SPIRV): -entry main -stage compute -target spirv -emit-spirv-directly
//TEST:SIMPLE(filecheck=HLSL): -entry main -stage compute -target hlsl
//TEST:SIMPLE(filecheck=METAL): -entry main -stage compute -target metal

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void main()
{
    float x = output[0];
    float3 vx = float3(x, x, x);

    float v1 = QuadReadLaneAt(x, 1);
    float v11 = QuadReadLaneAt(vx, 1).x;
    float v2 = QuadReadAcrossX(x);
    float v21 = QuadReadAcrossX(vx).x;
    float v3 = QuadReadAcrossY(x);
    float v31 = QuadReadAcrossY(vx).x;
    float v4 = QuadReadAcrossDiagonal(x);
    float v41 = QuadReadAcrossDiagonal(vx).x;

    output[0] = v1 + v2 + v3 + v4 + v11 + v21 + v31 + v41;

    // HLSL: QuadReadLaneAt
    // HLSL: QuadReadLaneAt
    // HLSL: QuadReadAcrossX
    // HLSL: QuadReadAcrossX
    // HLSL: QuadReadAcrossY
    // HLSL: QuadReadAcrossY
    // HLSL: QuadReadAcrossDiagonal
    // HLSL: QuadReadAcrossDiagonal

    // SPIRV: OpGroupNonUniformQuadBroadcast {{.*}} %{{u?int_3}} {{.*}} %{{u?int_1}}
    // SPIRV: OpGroupNonUniformQuadBroadcast {{.*}} %{{u?int_3}} {{.*}} %{{u?int_1}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_0}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_0}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_1}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_1}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_2}}
    // SPIRV: OpGroupNonUniformQuadSwap {{.*}} %{{u?int_3}} {{.*}} %{{u?int_2}}

    // METAL: quad_shuffle
    // METAL: quad_shuffle
    // METAL: ^ 1
    // METAL: quad_shuffle
    // METAL: quad_shuffle
    // METAL: ^ 2
    // METAL: quad_shuffle
    // METAL: quad_shuffle
    // METAL: ^ 3
    // METAL: quad_shuffle
    // METAL: quad_shuffle
}
