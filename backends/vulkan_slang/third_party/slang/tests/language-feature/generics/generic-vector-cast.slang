//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpEntryPoint

struct SomeStruct<Real : __BuiltinFloatingPointType> {
    vector<Real, 3> color;
    __init() {
        vector<float, 3> _color = vector<float, 3>(0);
        color = _color; // error 30019: expected an expression of type 'vector<Real,3>', got 'vector<float,3>'
        color = vector<Real, 3>(_color); // also does not work
    }
}

[numthreads(1, 1, 1)]
void computeMain() {
    SomeStruct<float> s;
}