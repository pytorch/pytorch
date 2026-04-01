//TEST:SIMPLE(filecheck=CHECK): -target spirv
struct ArrayLike<T>
{
}
__generic<T : IDifferentiable>
extension ArrayLike<T> : IDifferentiable
{
}

struct ExpectsDiff<T : IDifferentiable>
{
}

// test that using `ExpectsDiff<ArrayLike<T>>` is fine to the type system.
// this means that we must have checked the extension for `ArrayLike<T>` before
// we check the `t3` parameter for `test`.
void test<T : IDifferentiable>(ExpectsDiff<ArrayLike<T>> t3)
{
}

// CHECK: OpEntryPoint

[numthreads(1, 1, 1)]
void main()
{
    ExpectsDiff<ArrayLike<float>> v = {};
    test<float>(v);
}