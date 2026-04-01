//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry main

//WGSL-DAG: @builtin(sample_index)
//WGSL-DAG: @fragment

float4 main(uint sampleIndex : SV_SampleIndex) : SV_Target 
{
    return float4(1,1,1,1);
}
