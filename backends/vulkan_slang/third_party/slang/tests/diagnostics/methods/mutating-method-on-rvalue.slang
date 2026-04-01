// mutating-method-on-rvalue.slang

//DIAGNOSTIC_TEST:SIMPLE:-target hlsl -entry main

struct Counter
{
    int count;

    [mutating] void increment() { count++; }

    void bad()
    {
        increment();
    }
}

cbuffer C
{
    Counter gCounter;
}

[shader("compute")]
void main()
{
    gCounter.increment();
}