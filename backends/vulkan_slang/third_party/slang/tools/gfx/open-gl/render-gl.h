// render-d3d11.h
#pragma once

#include "../renderer-shared.h"

namespace gfx
{

SlangResult SLANG_MCALL createGLDevice(const IDevice::Desc* desc, IDevice** outDevice);

} // namespace gfx
