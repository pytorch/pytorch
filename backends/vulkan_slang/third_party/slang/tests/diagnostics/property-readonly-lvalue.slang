//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):
// use read-only property as l-value

struct MyType
{
    property float3 prop { get { return float3(1, 2, 3);}}
}

void test()
{
    MyType t;
    // CHECK: {{.*}}error 30047: argument passed to parameter '0' must be l-value.
    t.prop.x += 1.0;
}
