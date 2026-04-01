//TEST:SIMPLE(filecheck=CHECK): -stage fragment -target spirv -emit-spirv-directly -entry main

RWStructuredBuffer<float> output;

void main()
{
    {
        // CHECK: OpFwidth
        float w = 1.0;
        float b = fwidth(w);
        output[0] = b;

        // CHECK: OpDPdx
        float b1 = ddx(w);
        output[1] = b1;
        // CHECK: OpDPdy
        float b2 = ddy(w);
        output[2] = b2;

        // CHECK: OpFwidthCoarse
        float b3 = fwidth_coarse(w);
        output[3] = b3;

        // CHECK: OpDPdxCoarse
        float b4 = ddx_coarse(w);
        output[4] = b4;

        // CHECK: OpDPdyCoarse
        float b5 = ddy_coarse(w);
        output[5] = b5;

        // CHECK: OpFwidthFine
        float b6 = fwidth_fine(w);
        output[6] = b6;

        // CHECK: OpDPdxFine
        float b7 = ddx_fine(w);
        output[7] = b7;

        // CHECK: OpDPdyFine
        float b8 = ddy_fine(w);
        output[8] = b8;
    }

    {
        // CHECK: OpFwidth
        float3 w = 1.0;
        float3 b = fwidth(w);
        output[9] = b.x;

        // CHECK: OpDPdx
        float3 b1 = ddx(w);
        output[10] = b1.x;
        // CHECK: OpDPdy
        float3 b2 = ddy(w);
        output[11] = b2.x;

        // CHECK: OpFwidthCoarse
        float3 b3 = fwidth_coarse(w);
        output[12] = b3.x;

        // CHECK: OpDPdxCoarse
        float3 b4 = ddx_coarse(w);
        output[13] = b4.x;

        // CHECK: OpDPdyCoarse
        float3 b5 = ddy_coarse(w);
        output[14] = b5.x;

        // CHECK: OpFwidthFine
        float3 b6 = fwidth_fine(w);
        output[15] = b6.x;

        // CHECK: OpDPdxFine
        float3 b7 = ddx_fine(w);
        output[16] = b7.x;

        // CHECK: OpDPdyFine
        float3 b8 = ddy_fine(w);
        output[17] = b8.x;
    }
}
