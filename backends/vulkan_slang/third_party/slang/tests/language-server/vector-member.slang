//TEST:LANG_SERVER(filecheck=CHECK):
void f()
{
    float4 v;
//COMPLETE:6,7
    v.
}

// CHECK: x
// CHECK: y
// CHECK: z
// CHECK: w