// debug-vertex-layout.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugInputLayout : public DebugObject<IInputLayout>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IInputLayout* getInterface(const Slang::Guid& guid);
};

} // namespace debug
} // namespace gfx
