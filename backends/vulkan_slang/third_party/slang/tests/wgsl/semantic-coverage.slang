//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry main

//WGSL-DAG: @builtin(sample_mask)
//WGSL-DAG: @fragment

struct Out 
{
    uint coverage : SV_Coverage;
};

Out main() 
{
    Out output;
    output.coverage = 1;
    return output;
}
