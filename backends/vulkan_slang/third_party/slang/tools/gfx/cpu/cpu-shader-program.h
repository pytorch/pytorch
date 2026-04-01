// cpu-shader-program.h
#pragma once
#include "cpu-base.h"
#include "cpu-shader-object-layout.h"

namespace gfx
{
using namespace Slang;

namespace cpu
{

class ShaderProgramImpl : public ShaderProgramBase
{
public:
    RefPtr<RootShaderObjectLayoutImpl> layout;

    ~ShaderProgramImpl() {}
};

} // namespace cpu
} // namespace gfx
