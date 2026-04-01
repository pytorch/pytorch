// Tests automatic synthesis of Differential type requirement for generic types.
//
// This specifically tests a synthesis path that occurs when the lookup of the Differential type happens before the conformance-check.
// If this path doesn't construct the generic differential type correctly, it will throw an error when constructing the array
// in this line: Feature<3>.Differential b = {0.2, 0.3, 0.4};
// 

//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type
//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

__generic<let C : int> 
struct Feature: IDifferentiable
{
    float vals[C];
}


struct Linear<let C : int>
{
    typedef Feature<C> Input;
    typedef Feature<C> Output;

    [BackwardDerivative(eval_bwd)]
    Output eval(Input in_feature)
    {
        Output out_feature;
        for (int i = 0; i < C; i++)
        {
            out_feature.vals[i] = in_feature.vals[i] * 2.0;
        }
        return out_feature;
    }

    void eval_bwd(inout DifferentialPair<Input> in_feature_pair, Feature<C>.Differential d_output)
    {
        /* empty.. doesn't really matter */
    }
}

[Differentiable]
Feature<3> f(Feature<3> a, Linear<3> layer)
{
    return layer.eval(a);
}

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Feature<3> a = {1.0, 2.0, 3.0};
    Feature<3>.Differential b = {0.2, 0.3, 0.4};

    Linear<3> layer;

    var dpA = diffPair(a, b);

    var result = fwd_diff(f)(dpA, layer).d;

    outputBuffer[0] = result.vals[0];
    outputBuffer[1] = result.vals[1];
    outputBuffer[2] = result.vals[2];
}