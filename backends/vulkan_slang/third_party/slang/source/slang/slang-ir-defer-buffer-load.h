#pragma once

namespace Slang
{

/*
This pass implements a targeted optimization that defers the loading of structured buffer elements
to the end of the access chain to avoid loading and repacking unnecessary data.
For example, if we see:
    val = StructuredBufferLoad(s, i)
    val2 = GetElement(val, j)
    val3 = FieldExtract(val2, field_key_0)
    call(foo, val3)
We should rewrite the code into:
    ptr = RWStructuredBufferGetElementPtr(s, i)
    ptr2 = ElementAddress(ptr, j)
    ptr3 = FieldAddress(ptr2, field_key_0)
    val3 = Load(ptr3)
    call(foo, val3)
*/

struct IRModule;

void deferBufferLoad(IRModule* module);

} // namespace Slang
