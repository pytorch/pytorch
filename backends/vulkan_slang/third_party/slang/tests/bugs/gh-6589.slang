//TEST:SIMPLE(filecheck=CHECK): -target spirv

// This is a test that checks that we can apply partial specialization to a function
// we won't specialize the function parameters too aggressively. Instead, we will specialize
// the parameters at the same time of specializing the arguments. Otherwise, we could lose
// the chance to specialize the argument.
//
// In this test, `matrix_vector_interfaces` will be fully specialized, otherwise the compile
// will fail because we don't allow opaque type in the existential type. So as long as the target
// spirv code can be generated, we are good.

// CHECK: %main
public interface ITensor<T : IDifferentiable, let D : int>
{
    public T get(int idx);

}

public interface IRWTensor<T : IDifferentiable, let D : int> : ITensor<T, D>
{
}


public struct RWTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    public RWStructuredBuffer<T> buffer;
    public T get(int idx) { return buffer[idx]; }
}

public struct GradInOutTensor<T : IDifferentiable, let D : int> : IRWTensor<T, D>
{
    public RWTensor<T, D> primal;
    public T get(int idx) { return primal.get(idx); }
}

struct CallData
{
    GradInOutTensor<float, 3> weights;
    GradInOutTensor<float, 2> biases;
    RWStructuredBuffer<float> _result;
}
ParameterBlock<CallData> call_data;

float matrix_vector_interfaces(ITensor<float, 2> weights, ITensor<float, 1> biases)
{
    return weights.get(0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    float _result;
    GradInOutTensor<float, 2> weights;
    GradInOutTensor<float, 1> biases;

    weights.primal.buffer = call_data.weights.primal.buffer;
    biases.primal.buffer = call_data.biases.primal.buffer;

    _result = matrix_vector_interfaces(weights, biases);

    call_data._result[0] = _result;
}
