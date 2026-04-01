//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* Tests around interface/generics

.slang(7): error 20001: unexpected '<', expected '{'
interface IThing<T>
*/

// Docs say this should work...
// https://github.com/shader-slang/slang/blob/master/docs/language-reference/07-declarations.md

interface IThing<T>
{
    T get();
};

struct X : IThing<int>
{
    int get() { return 10; }
};

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    int index = dispatchThreadID.x;

    X x;
    IThing<int> i = x;
    
	outputBuffer[index] = i.get();
}