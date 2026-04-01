//TEST:LANG_SERVER(filecheck=CHECK):
//
__generic < T : struct> extension T
{
// CHECK: null
//HOVER:7,32
    __init(StructuredBuffer<T>) {}
}