// mismatching-types.slang
//DIAGNOSTIC_TEST:SIMPLE:-target hlsl

Texture1D<float> tex;

struct GenericOuter<T>
{
    struct GenericInner<Q>
    {
        T val;
    };

    struct NonGenericInner
    {
        T val;
    };

    GenericInner<T> g;
    NonGenericInner ng;
};

struct NonGenericOuter
{
    struct GenericInner<T>
    {
        T val;

        struct ReallyNested
        {
            T val;
        };

        ReallyNested n;
    };

    GenericInner<int> i;
    GenericInner<float> f;
};


[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{    
    GenericOuter<int> a;
    GenericOuter<float> b;
    NonGenericOuter c;
    NonGenericOuter.GenericInner<int> d;
    
    // expected an expression of type 'GenericOuter<int>', got 'int'
    a = 0;

    // expected an expression of type 'GenericOuter<int>.GenericInner<int>', got 'int'
    // explicit conversion from 'int' to 'GenericOuter<int>.GenericInner<int>' is possible
    a.g = 0;

    // expected an expression of type 'GenericOuter<int>.NonGenericInner', got 'int'
    // explicit conversion from 'int' to 'GenericOuter<int>.GenericInner<int>' is possible
    a.ng = 0;
    // expected an expression of type 'GenericOuter<int>.GenericInner<int>', got 'GenericOuter<float>.GenericInner<float>'
    a.g = b.g;
    // expected an expression of type 'GenericOuter<int>.NonGenericInner', got 'GenericOuter<float>.NonGenericInner'
    a.ng = b.ng;
    // expected an expression of type 'NonGenericOuter.GenericInner<int>', got 'int'
    // explicit conversion from 'int' to 'GenericInner<int>' is possible
    c.i = 0;
    // expected an expression of type 'NonGenericOuter.GenericInner<int>', got 'NonGenericOuter.GenericInner<float>'
    c.i = c.f;

    // expected an expression of type 'NonGenericOuter.GenericInner<int>.ReallyNested', got 'int'
    // explicit conversion from 'int' to 'GenericInner<int>.ReallyNested' is possible
    c.i.n = 0;
    // OK
    c.i.n.val = 0;
    // OK
    d.val = 0;
    // OK
    c.i = d;

    // expected an expression of type 'Texture1D<int>', got 'Texture1D<float>'
    Texture1D<int> t1 = tex;
    // expected an expression of type 'Texture2D<float>', got 'Texture1D<float>'
    Texture2D<float> t2 = tex;
}
