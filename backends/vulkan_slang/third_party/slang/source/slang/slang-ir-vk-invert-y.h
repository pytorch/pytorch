#ifndef SLANG_IR_VK_INVERT_Y_H
#define SLANG_IR_VK_INVERT_Y_H

namespace Slang
{
struct IRModule;
void invertYOfPositionOutput(IRModule* module);
void rcpWOfPositionInput(IRModule* module);
} // namespace Slang

#endif
