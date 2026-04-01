//DISABLE_TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj

/* The purpose of these tests is to use generics to be able to traverse
a 'linked list' of different related types. 

tests/current-bugs/generic/byte-address-ptr.slang(77): note 99999: an internal error threw an exception while working on code near this location
(0): error 99999: Slang compilation aborted due to an exception of class Slang::InternalError: assert failure: witnessTableVal->getOp() != kIROp_StructKey
*/

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name outputBuffer

//TEST_INPUT:ubuffer(data=[0]):name=gB
ByteAddressBuffer gB;

enum Type : int
{
    A,
    B,
    C,
};

interface IBase 
{
    static bool isType(Type type);
};

struct Base : IBase
{
    static bool isType(Type type) { return true; }
    Type type;
};

struct Ptr
{
    Type getType() { return gB.Load<Base>(offset).type; }
    
    __generic<T : IBase>
    bool isType() { return T::isType(getType()); }
    // Changing to T : IBase causes an internal compiler crash
    __generic<T : IBase>
    T as()
    {
        bool isOk = isType<T>();
        // I want a way to assert!
        // assert(isOk);
        return gB.Load<T>(offset);
    }
    int offset;
};

struct A : Base
{
    static bool isType(Type type) { return type == Type::A; }
    Ptr next;
    int something;
};

struct B : Base
{
    static bool isType(Type type) { return type == Type::C || type == Type::B; }
    int someData;
};

struct C : B
{
    static bool isType(Type type) { return type == Type::C; }
    int somethingElse;
};

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    Ptr ptr = { 0 };
    int total = 0;
    
    while (ptr.isType<A>())    
    {
        A a = ptr.as<A>();
        total += a.something;

        ptr = a.next;
    }

    outputBuffer[dispatchThreadID.x] = total;
}