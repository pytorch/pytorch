//TEST:SIMPLE:

// It used to be the case that coercions of already-coerced initializer lists
// didn't take into account the more specific type. One way of triggering this
// was to create an already-coerced initializer list via the HLSL `(S)0`
// syntax which creates the equivalent of `{} : S`. In the example below this
// then proceeds to be coerced to type T.

struct S {};
struct T {};

void main()
{
    // Should generate a type error.
    T t = (S)0;
}

