//TEST:LANG_SERVER(filecheck=CHECK):

float3 m()
{
    float3 val = 0;
//COMPLETE:7,9
    val.
    return val;
}

// CHECK: x
// CHECK: y
// CHECK: z