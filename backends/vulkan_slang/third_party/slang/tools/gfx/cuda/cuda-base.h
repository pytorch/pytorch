// cuda-base.h
// Shared header file for CUDA implementation
#pragma once

#ifdef GFX_ENABLE_CUDA
#include "../command-encoder-com-forward.h"
#include "../command-writer.h"
#include "../mutable-shader-object.h"
#include "../renderer-shared.h"
#include "../simple-transient-resource-heap.h"
#include "../slang-context.h"
#include "core/slang-basic.h"
#include "core/slang-blob.h"
#include "core/slang-std-writers.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <cuda.h>

#ifdef RENDER_TEST_OPTIX

// The `optix_stubs.h` header produces warnings when compiled with MSVC
#ifdef _MSC_VER
#pragma warning(disable : 4996)
#endif

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#endif

#endif

namespace gfx
{
namespace cuda
{
class CUDAContext;
class BufferResourceImpl;
class TextureResourceImpl;
class ResourceViewImpl;
class ShaderObjectLayoutImpl;
class RootShaderObjectLayoutImpl;
class ShaderObjectImpl;
class MutableShaderObjectImpl;
class EntryPointShaderObjectImpl;
class RootShaderObjectImpl;
class ShaderProgramImpl;
class PipelineStateImpl;
class QueryPoolImpl;
class DeviceImpl;
class CommandBufferImpl;
class ResourceCommandEncoderImpl;
class ComputeCommandEncoderImpl;
class CommandQueueImpl;
} // namespace cuda
} // namespace gfx
