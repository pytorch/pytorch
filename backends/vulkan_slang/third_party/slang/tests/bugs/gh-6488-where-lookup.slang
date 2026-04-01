//TEST:SIMPLE(filecheck=CHECK): -target spirv

// CHECK: OpEntryPoint

interface IRealArray<T : __BuiltinFloatingPointType, int D> {}
extension<T : __BuiltinFloatingPointType, int D> vector<T, D> : IRealArray<T, D> where T.Differential : __BuiltinFloatingPointType { }


[numthreads(1,1,1)]
void main()
{}