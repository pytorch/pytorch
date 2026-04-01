//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* A test around use of an array like container. 

Here we just fix the array size, so can test out other characteristics.

indexOf can't compile because v == elements[i] cannot determine equality.

It would seem like I should have an IEquality interface, that I could then make
a T require. 

Doesn't work because 

.slang(29): error 30019: expected an expression of type 'Type', got 'T'
            if (T::isEqual(v, elements[i]))

 */

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer
RWStructuredBuffer<float> outputBuffer;

interface IEquality
{
    associatedtype Type;
    static bool isEqual(Type a, Type b);
};

extension int : IEquality
{
    typedef int Type;	
    static bool isEqual(Type a, Type b)
    {
        return a == b;
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
            if (T::isEqual(v, elements[i]))
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


