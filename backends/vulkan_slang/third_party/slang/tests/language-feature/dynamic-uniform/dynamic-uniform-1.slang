

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK): -target hlsl -entry main -profile cs_6_0 -validate-uniformity

RWStructuredBuffer<int> output;
void expectUniform(dynamic_uniform int v)
{
    output[0] = v;
}

[NonUniformReturn]
int produceNonUniform() { return 0; }

struct B
{
    int a;
    int c;
    dynamic_uniform int d;
    dynamic_uniform int d1;
    dynamic_uniform int d2;
    dynamic_uniform int d3;
    dynamic_uniform int d4;
}

int getBa(B obj)
{
    return obj.a;
}

int getBaFromInout(inout B obj)
{
    return obj.a;
}

int getBdFromInout(inout B obj)
{
    return obj.d;
}

ConstantBuffer<B> ub;
void main(int tid: SV_DispatchThreadID)
{
    int d = ub.a + ub.c;
    int f = ub.a + tid;
    int ff = ub.a + produceNonUniform();
    
    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    expectUniform(f);
    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    expectUniform(ff);

    B obj = ub;
    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    expectUniform(getBaFromInout(obj));
    obj.a = ff;
    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    expectUniform(getBa(obj));

    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    obj.d = ff;

    int t = 0;
    if (tid == 0)
        t = 1;

    // CHECK-DAG: ([[# @LINE+1]]): warning 41201
    obj.d1 = t;

    int u = 0;
    if (ub.a == 0)
        u = 1;
    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    obj.d2 = u;

    int vv = 0;
    if (tid == 0)
        vv = 2;
    else
        vv = 2;
    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    obj.d3 = vv;

    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    expectUniform(getBdFromInout(obj));
    
    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    obj.d4 = d; // OK

    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    expectUniform(d);

    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    expectUniform(getBa(ub));

    // CHECK-NOT: ([[# @LINE+1]]): warning 41201
    expectUniform(asDynamicUniform(ff));

}
