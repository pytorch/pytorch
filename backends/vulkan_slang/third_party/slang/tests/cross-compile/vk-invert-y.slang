// vk-invert-y.slang

// Test to confirm that '-f-vk-invert-y' option works as expected.

//TEST:SIMPLE(filecheck=CHECK):-target glsl -entry main -stage vertex -profile vs_5_1 -fvk-invert-y

struct VOutput
{
    float4 v : SV_Position;
}


VOutput main()
{
    VOutput output;
    output.v = float4(1, 2, 3, 4);
    
    // CHECK: [[TMP:[_A-Za-z0-9]+]].y = - output_0.v_0.y;
    // CHECK: gl_Position = [[TMP]];
    return output;
}
