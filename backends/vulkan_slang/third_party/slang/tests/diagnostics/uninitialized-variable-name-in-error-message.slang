//TEST:SIMPLE(filecheck=CHK):

// Test if the variable name is a part of the error message
// when it is used inside of for-loop

RWStructuredBuffer<float> gInput;
RWStructuredBuffer<float> outputBuffer;

//CHK-DAG: ([[#@LINE+1]]): warning 41016: use of uninitialized variable 'a'
float func1() { float a; return a; }

//CHK-DAG: ([[#@LINE+1]]): warning 41016: use of uninitialized variable 'b'
float func2() { float b; return b; }

int test(int inVal)
{
    return inVal;
}

[Shader("compute")]
[NumThreads(4, 1, 1)]
void computeMain(int3 dispatchThreadID : SV_DispatchThreadID)
{
    int tid = dispatchThreadID.x;
    int inVal1; // intentionally uninitialized
    int inVal2; // intentionally uninitialized

    for (int i = 0; i <2; ++i)
    {
        // CHK-DAG: ([[#@LINE+1]]): warning 41016: use of uninitialized variable 'inVal1'
        int outVal = test(inVal1);

        // CHK-DAG: ([[#@LINE+1]]): warning 41016: use of uninitialized variable 'inVal2'
        outVal += test(inVal2);
        outputBuffer[tid] = outVal;
    }
}
