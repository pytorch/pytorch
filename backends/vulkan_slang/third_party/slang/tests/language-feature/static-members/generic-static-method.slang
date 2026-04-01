// generic-static-method.slang

// Confirm that the compiler can handle a generic
// `static` method declaration and call.

//TEST:SIMPLE:

interface IFrobnicator
{
    float frobnicate(float value);
}

struct Doubler : IFrobnicator
{
    float frobnicate(float value) { return 2.0f * value; }
}

struct FrobnicateHelpers
{
    static float doubleFrobnicate<F : IFrobnicator>(F f, float value)
    {
        return f.frobnicate(f.frobnicate(value));
    }
}

float test(float value)
{
    Doubler d;
    return FrobnicateHelpers.doubleFrobnicate(d, value);
}
