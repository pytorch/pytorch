// mutating-impl-of-non-mutating-req.slang

//DIAGNOSTIC_TEST:SIMPLE(filecheck=CHECK):-target hlsl -entry main
interface IThing
{
    int processValue(int inValue);
}

struct Counter : IThing
{
    int state;
    
    // CHECK: ([[# @LINE+1]]): error 38105:
    [mutating] int processValue(int inValue)
    {
        int result = state;
        state += inValue;
        return state;
    }
}

int helper<T : IThing>(T thing, int value)
{
    return thing.processValue(value);
}

int test(int value)
{
    Counter counter = { value };
    return helper(counter, value);
}

cbuffer C
{
    int gValue;
}

[shader("fragment")]
int main() : SV_Target
{
    return test(gValue);
}