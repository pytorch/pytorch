// slang-extension-tracker.cpp
#include "slang-extension-tracker.h"

namespace Slang
{

void ShaderExtensionTracker::appendExtensionRequireLinesForGLSL(StringBuilder& ioBuilder) const
{
    for (const auto& extension : m_extensionPool.getSlices())
    {
        ioBuilder.append("#extension ");
        ioBuilder.append(extension);
        ioBuilder.append(" : require\n");
    }
}

void ShaderExtensionTracker::appendExtensionRequireLinesForWGSL(StringBuilder& ioBuilder) const
{
    for (const auto& extension : m_extensionPool.getSlices())
    {
        ioBuilder.append("enable ");
        ioBuilder.append(extension);
        ioBuilder.append(";\n");
    }
}

void ShaderExtensionTracker::requireSPIRVVersion(const SemanticVersion& version)
{
    if (version > m_spirvVersion)
    {
        m_spirvVersion = version;
    }
}

void ShaderExtensionTracker::requireVersion(ProfileVersion version)
{
    // Check if this profile is newer
    if ((UInt)version > (UInt)m_profileVersion)
    {
        m_profileVersion = version;
    }
}

void ShaderExtensionTracker::requireBaseTypeExtension(BaseType baseType)
{
    uint32_t bit = 1 << int(baseType);
    if (m_hasBaseTypeFlags & bit)
    {
        return;
    }

    switch (baseType)
    {
    case BaseType::UInt8:
    case BaseType::Int8:
        {
            // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_8bit_storage.txt
            requireExtension(UnownedStringSlice::fromLiteral("GL_EXT_shader_8bit_storage"));

            // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt
            requireExtension(
                UnownedStringSlice::fromLiteral("GL_EXT_shader_explicit_arithmetic_types"));
            break;
        }
    case BaseType::Half:
    case BaseType::UInt16:
    case BaseType::Int16:
        {
            // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_16bit_storage.txt
            requireExtension(UnownedStringSlice::fromLiteral("GL_EXT_shader_16bit_storage"));

            // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt
            requireExtension(
                UnownedStringSlice::fromLiteral("GL_EXT_shader_explicit_arithmetic_types"));
            break;
        }
    case BaseType::UInt64:
    case BaseType::Int64:
        {
            requireExtension(
                UnownedStringSlice::fromLiteral("GL_EXT_shader_explicit_arithmetic_types_int64"));
            m_hasBaseTypeFlags |= _getFlag(BaseType::UInt64) | _getFlag(BaseType::Int64) |
                                  _getFlag(BaseType::IntPtr) | _getFlag(BaseType::UIntPtr);
            break;
        }
    case BaseType::IntPtr:
    case BaseType::UIntPtr:
        {
#if SLANG_PTR_IS_64
            requireExtension(
                UnownedStringSlice::fromLiteral("GL_EXT_shader_explicit_arithmetic_types_int64"));
            m_hasBaseTypeFlags |= _getFlag(BaseType::UInt64) | _getFlag(BaseType::Int64) |
                                  _getFlag(BaseType::IntPtr) | _getFlag(BaseType::UIntPtr);
#endif
            break;
        }
    }

    m_hasBaseTypeFlags |= bit;
}

} // namespace Slang
