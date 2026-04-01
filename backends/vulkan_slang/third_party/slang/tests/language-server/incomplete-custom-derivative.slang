
//TEST:LANG_SERVER(filecheck=CHECK):

RWByteAddressBuffer atomicBuffer;
RWByteAddressBuffer atomicGradBuffer;

float myDifferentiableMin(uint location, float value)
{
    uint originalValue;
    atomicBuffer.InterlockedMin(location, asuint(value), originalValue);
//SIGNATURE:12,20
    InterlockedMin()
    return min(asfloat(originalValue), value);
}
//HOVER:17,12
[BackwardDerivativeOf(myDifferentiableMin)]
float myDiffMin()

// CHECK: func InterlockedMin
// CHECK: func myDiffMin() -> float
