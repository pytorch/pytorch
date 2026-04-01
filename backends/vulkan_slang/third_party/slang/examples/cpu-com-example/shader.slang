// shader.slang

// Example using 'NativeString' and COM interface

export __global __extern_cpp IDoThings globalDoThings;

export __extern_cpp NativeString getString(NativeString in)
{
    return in;
}

[COM("007E2DAF-3A1F-4BC3-B08B-A9DB233D38CE")]
interface IDoThings
{
    int doThing(int a, int b);
    int calcHash(NativeString in);
    void printMessage(NativeString nativeString);
}

export __extern_cpp int calcHash(NativeString text)
{
    return globalDoThings.calcHash(text);
}

export __extern_cpp void printMessage(NativeString text)
{
    return globalDoThings.printMessage(text);
}
