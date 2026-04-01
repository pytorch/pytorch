//TEST:LANG_SERVER(filecheck=CHECK):
void f()
{
    float v;
//COMPLETE:6,7
    v.
}

// CHECK: x