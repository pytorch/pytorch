//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

struct Type
{
    int member;
}

struct CB
{
    Type* ptr;
}
ConstantBuffer<CB> cb;

[numthreads(1,1,1)]
void main()
{
    Type val;

    // CHECK: ([[# @LINE+1]]): error 30101
    val->member = 2; // Error.

    // CHECK-NOT: error
    let a = cb->ptr->member; // OK.

}