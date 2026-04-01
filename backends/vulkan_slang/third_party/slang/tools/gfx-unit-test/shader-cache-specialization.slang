// shader-cache-specialization.slang

// This is a copy of `shader-object.slang` in `shader-object` example
// for use by compute-smoke gfx unit test.

// This file implements a simple compute shader that transforms
// input floating point numbers stored in a `RWStructuredBuffer`.
// Specifically, for each number x from input buffer, compute
// f(x) and store the result back in the same buffer.

// The compute shader supports multiple transformation functions,
// such add(x, c) which returns x+c, or mul(x, c) which returns x*c.
// This functions are implemented as types that conforms to the
// `ITransformer` interface.

// The main entry point function takes a parameter of `ITransformer`
// type, and applies the transformation to numbers in the input
// buffer. By defining the shader parameter using interfaces,
// we enable the flexiblity to generate either specialized compute
// kernels that performs specific transformation or a general
// kernel that can perform any transformations encoded by the
// parameter at run-time, without changing any shader code or
// host-application logic for setting and preparing shader parameters.

// Defines the transformer interface, which implements a single
// `transform` operation.
interface ITransformer
{
    float transform(float x);
}

// Represents a transform function f(x) = x + c.
struct AddTransformer : ITransformer
{
    float c;
    float transform(float x) { return x + c; }
};

// Represents a transform function f(x) = x * c.
struct MulTransformer : ITransformer
{
    float c;
    float transform(float x) { return x * c; }
};

// Represents a composite function f(x) = f0(f1(x));
struct CompositeTransformer : ITransformer
{
    ITransformer func0;
    ITransformer func1;
    float transform(float x)
    {
        return func0.transform(func1.transform(x));
    }
};

// Main entry-point. Applies the transformation encoded by `transformer`
// to all elements in `buffer`.
[shader("compute")]
[numthreads(4,1,1)]
void computeMain(
    uint3 sv_dispatchThreadID : SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> buffer,
    uniform ITransformer transformer)
{
    var input = buffer[sv_dispatchThreadID.x];
    buffer[sv_dispatchThreadID.x] = transformer.transform(input);
}
