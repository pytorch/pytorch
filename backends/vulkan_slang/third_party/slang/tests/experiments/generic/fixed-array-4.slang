//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test around use of an array like container. 
 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

/* Here is an attempt to provide an interface to check equality. 
But is the definition right anyway? Really I want the This and Type to be 
the same type. I guess this enforces that but in an odd manner.
*/
interface IEquality
{
    associatedtype Type;
    bool isEqual(Type rhs);
};

extension int : IEquality
{
    typedef int Type;
    bool isEqual(Type rhs)
    {
        // This use of `this` is going to seem odd to a C++ programmer 
        return this == rhs;
    }
};

struct FixedArray<T : IEquality> 
{
    static const int SIZE = 4;
    
    [mutating] void setAt(int i, T value) { elements[i] = value; }
    T getAt(int i) { return elements[i]; }
    int indexOf(T v)
    {
        for (int i = 0; i < SIZE; ++i)
        {
            if (v.isEqual(elements[i]))
            {
                return i;
            }
            return -1;
        }
    }
    T elements[SIZE];
};


[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = dispatchThreadID.x;

    FixedArray<int> arr;
    arr.setAt(0, index);
    
	outputBuffer[index] = 1 + arr.getAt(0);
}

