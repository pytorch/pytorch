//TEST_DISABLED:EVAL:

// Note: This test has been disabled as part of introducing
// the IR-level type system, because it changes the overall
// structure of IR moduels quite a bit, and no user code
// actually relies on the serialized IR or VM.
//
// This test should ideally be re-enabled once work is
// done to revamp the serialized bytecode format into
// something more essential to the compiler (e.g., for
// modular separate compilation).

StructuredBuffer<int> input;
RWStructuredBuffer<int> output;

int factorial(int n)
{
    int result = 1;
    while(n > 0)
    {
        result *= n;
        n--;
    }
    return result;
}

[numthreads(1, 1, 1)]
void main(
    uint tid   : SV_DispatchThreadID)
{
    output[tid] = factorial(input[tid]);
}

