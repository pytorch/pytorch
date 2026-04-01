#include "parameter-recorder.h"

namespace SlangRecord
{
void ParameterRecorder::recordStruct(SlangGlobalSessionDesc const& desc)
{
    recordUint32(desc.structureSize);
    recordUint32(desc.apiVersion);
    recordUint32(desc.languageVersion);
    recordUint32(desc.enableGLSL);
}

void ParameterRecorder::recordStruct(slang::SessionDesc const& desc)
{
    recordUint64(desc.structureSize);
    recordInt64(desc.targetCount);

    for (SlangInt i = 0; i < desc.targetCount; i++)
    {
        recordStruct(desc.targets[i]);
    }

    recordUint32(desc.flags);
    recordEnumValue(desc.defaultMatrixLayoutMode);
    recordInt64(desc.searchPathCount);
    for (SlangInt i = 0; i < desc.searchPathCount; i++)
    {
        recordString(desc.searchPaths[i]);
    }

    recordInt64(desc.preprocessorMacroCount);
    for (SlangInt i = 0; i < desc.preprocessorMacroCount; i++)
    {
        recordStruct(desc.preprocessorMacros[i]);
    }

    recordBool(desc.enableEffectAnnotations);
    recordBool(desc.allowGLSLSyntax);

    recordUint32(desc.compilerOptionEntryCount);
    for (uint32_t i = 0; i < desc.compilerOptionEntryCount; i++)
    {
        recordStruct(desc.compilerOptionEntries[i]);
    }
}

void ParameterRecorder::recordStruct(slang::PreprocessorMacroDesc const& desc)
{
    recordString(desc.name);
    recordString(desc.value);
}

void ParameterRecorder::recordStruct(slang::CompilerOptionEntry const& entry)
{
    recordEnumValue(entry.name);
    recordStruct(entry.value);
}

void ParameterRecorder::recordStruct(slang::CompilerOptionValue const& value)
{
    recordEnumValue(value.kind);
    recordInt32(value.intValue0);
    recordString(value.stringValue0);
    recordString(value.stringValue1);
}

void ParameterRecorder::recordStruct(slang::TargetDesc const& targetDesc)
{
    recordUint64(targetDesc.structureSize);
    recordEnumValue(targetDesc.format);
    recordEnumValue(targetDesc.profile);
    recordEnumValue(targetDesc.flags);
    recordEnumValue(targetDesc.floatingPointMode);
    recordEnumValue(targetDesc.lineDirectiveMode);
    recordBool(targetDesc.forceGLSLScalarBufferLayout);
    recordUint32(targetDesc.compilerOptionEntryCount);
    for (uint32_t i = 0; i < targetDesc.compilerOptionEntryCount; i++)
    {
        recordStruct(targetDesc.compilerOptionEntries[i]);
    }
}

void ParameterRecorder::recordStruct(slang::SpecializationArg const& specializationArg)
{
    recordEnumValue(specializationArg.kind);
    recordAddress(specializationArg.type);
}

void ParameterRecorder::recordPointer(const void* value, bool omitData, size_t size)
{
    recordAddress(value);
    if (omitData)
    {
        recordUint64(0llu);
        return;
    }

    recordUint64(size);
    if (size)
    {
        m_stream->write(value, size);
    }
}

void ParameterRecorder::recordPointer(ISlangBlob* blob)
{
    recordAddress(static_cast<const void*>(blob));

    if (blob)
    {
        size_t size = blob->getBufferSize();
        const void* buffer = blob->getBufferPointer();
        recordPointer(buffer, false, size);
    }
}

// first 4-bytes is the length of the string
void ParameterRecorder::recordString(const char* value)
{
    if (value == nullptr)
    {
        recordUint32(0);
    }
    else
    {
        uint32_t size = (uint32_t)strlen(value);
        recordUint32(size);
        m_stream->write(value, size);
    }
}
} // namespace SlangRecord
