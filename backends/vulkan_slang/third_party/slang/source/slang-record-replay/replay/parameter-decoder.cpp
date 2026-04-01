#include "parameter-decoder.h"

#include <string.h>

namespace SlangRecord
{
size_t ParameterDecoder::decodeString(
    const uint8_t* buffer,
    int64_t bufferSize,
    PointerDecoder<char*>& typeDecoder)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    if (bufferSize < (int64_t)sizeof(uint32_t))
    {
        return 0;
    }

    uint32_t stringLength = 0;
    size_t readByte = 0;
    readByte += decodeUint32(buffer, bufferSize - readByte, stringLength);

    SLANG_RECORD_ASSERT(bufferSize >= (int64_t)(readByte + stringLength));

    if (stringLength == 0)
    {
        return readByte;
    }

    uint8_t* data = (uint8_t*)typeDecoder.allocate(stringLength + 1);
    memcpy(data, buffer + readByte, stringLength);
    typeDecoder.setPointer(data);
    typeDecoder.setDataSize(stringLength + 1);
    return readByte + stringLength;
}

size_t ParameterDecoder::decodePointer(
    const uint8_t* buffer,
    int64_t bufferSize,
    PointerDecoder<void*>& pointerDecoder)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    if (bufferSize < (int64_t)sizeof(uint32_t))
    {
        return 0;
    }

    uint64_t address = 0;
    size_t readByte = decodeAddress(buffer, bufferSize, address);
    pointerDecoder.setPointerAddress(address);

    uint64_t dataSize = 0;
    readByte += decodeUint64(buffer + readByte, bufferSize - readByte, dataSize);

    // return if the data size is 0
    if (dataSize == 0)
    {
        return readByte;
    }

    SLANG_RECORD_ASSERT(bufferSize >= (int64_t)(readByte + dataSize));

    uint8_t* data = (uint8_t*)pointerDecoder.allocate(dataSize);
    memcpy(data, buffer + readByte, dataSize);
    pointerDecoder.setPointer(data);
    pointerDecoder.setDataSize(dataSize);
    return readByte + dataSize;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<SlangGlobalSessionDesc>& sessionDesc)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    if (bufferSize < (int64_t)sizeof(uint64_t))
    {
        return 0;
    }

    size_t readByte = 0;
    SlangGlobalSessionDesc& desc = sessionDesc.getValue();
    readByte = decodeUint32(buffer, bufferSize, desc.structureSize);
    readByte += decodeUint32(buffer + readByte, bufferSize - readByte, desc.apiVersion);
    readByte += decodeUint32(buffer + readByte, bufferSize - readByte, desc.languageVersion);
    uint32_t val = 0;
    readByte += decodeUint32(buffer + readByte, bufferSize - readByte, val);
    desc.enableGLSL = (val != 0);

    return readByte;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::SessionDesc>& sessionDesc)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    if (bufferSize < (int64_t)sizeof(uint64_t))
    {
        return 0;
    }

    size_t readByte = 0;
    slang::SessionDesc& desc = sessionDesc.getValue();

    uint64_t structSize = 0;
    readByte = decodeUint64(buffer, bufferSize, structSize);
    desc.structureSize = structSize;

    readByte += decodeInt64(buffer + readByte, bufferSize - readByte, desc.targetCount);

    if (desc.targetCount > 0)
    {
        slang::TargetDesc* targets =
            (slang::TargetDesc*)sessionDesc.allocate(sizeof(slang::TargetDesc) * desc.targetCount);
        readByte +=
            decodeStructArray(buffer + readByte, bufferSize - readByte, targets, desc.targetCount);
        desc.targets = targets;
    }

    readByte += decodeUint32(buffer + readByte, bufferSize - readByte, desc.flags);
    readByte +=
        decodeEnumValue(buffer + readByte, bufferSize - readByte, desc.defaultMatrixLayoutMode);
    readByte += decodeInt64(buffer + readByte, bufferSize - readByte, desc.searchPathCount);

    if (desc.searchPathCount > 0)
    {
        char** searchPaths = (char**)sessionDesc.allocate(sizeof(char*) * desc.searchPathCount);
        decodeStringArray(
            buffer + readByte,
            bufferSize - readByte,
            searchPaths,
            desc.searchPathCount);
        desc.searchPaths = searchPaths;
    }

    readByte += decodeInt64(buffer + readByte, bufferSize - readByte, desc.preprocessorMacroCount);
    if (desc.preprocessorMacroCount > 0)
    {
        slang::PreprocessorMacroDesc* macros = (slang::PreprocessorMacroDesc*)sessionDesc.allocate(
            sizeof(slang::PreprocessorMacroDesc) * desc.preprocessorMacroCount);
        readByte += decodeStructArray(
            buffer + readByte,
            bufferSize - readByte,
            macros,
            desc.preprocessorMacroCount);
        desc.preprocessorMacros = macros;
    }

    readByte += decodeBool(buffer + readByte, bufferSize - readByte, desc.enableEffectAnnotations);
    readByte += decodeBool(buffer + readByte, bufferSize - readByte, desc.allowGLSLSyntax);
    readByte +=
        decodeUint32(buffer + readByte, bufferSize - readByte, desc.compilerOptionEntryCount);

    if (desc.compilerOptionEntryCount > 0)
    {
        slang::CompilerOptionEntry* entries = (slang::CompilerOptionEntry*)sessionDesc.allocate(
            sizeof(slang::CompilerOptionEntry) * desc.compilerOptionEntryCount);
        readByte += decodeStructArray(
            buffer + readByte,
            bufferSize - readByte,
            entries,
            desc.compilerOptionEntryCount);
        desc.compilerOptionEntries = entries;
    }

    return readByte;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::PreprocessorMacroDesc>& desc)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    size_t readByte = 0;
    PointerDecoder<char*> name;
    PointerDecoder<char*> value;

    readByte = decodeString(buffer, bufferSize, name);
    readByte += decodeString(buffer + readByte, bufferSize - readByte, value);

    desc.getValue().name = name.getPointer();
    desc.getValue().value = value.getPointer();

    return readByte;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::CompilerOptionEntry>& entry)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    size_t readByte = 0;
    readByte = decodeEnumValue(buffer, bufferSize, entry.getValue().name);

    ValueDecoder<slang::CompilerOptionValue> value;
    readByte += decodeStruct(buffer + readByte, bufferSize - readByte, value);
    entry.getValue().value = value.getValue();

    return readByte;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::CompilerOptionValue>& value)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    size_t readByte = 0;
    readByte = decodeEnumValue(buffer, bufferSize, value.getValue().kind);
    readByte += decodeInt32(buffer + readByte, bufferSize - readByte, value.getValue().intValue0);

    PointerDecoder<char*> stringValue0;
    readByte += decodeString(buffer + readByte, bufferSize - readByte, stringValue0);
    value.getValue().stringValue0 = stringValue0.getPointer();

    PointerDecoder<char*> stringValue1;
    readByte += decodeString(buffer + readByte, bufferSize - readByte, stringValue1);
    value.getValue().stringValue1 = stringValue1.getPointer();
    return 0;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::TargetDesc>& targetDesc)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    size_t readByte = 0;
    uint64_t structSize = 0;
    readByte = decodeUint64(buffer, bufferSize, structSize);
    targetDesc.getValue().structureSize = structSize;

    readByte +=
        decodeEnumValue(buffer + readByte, bufferSize - readByte, targetDesc.getValue().format);
    readByte +=
        decodeEnumValue(buffer + readByte, bufferSize - readByte, targetDesc.getValue().profile);
    readByte +=
        decodeEnumValue(buffer + readByte, bufferSize - readByte, targetDesc.getValue().flags);
    readByte += decodeEnumValue(
        buffer + readByte,
        bufferSize - readByte,
        targetDesc.getValue().floatingPointMode);
    readByte += decodeEnumValue(
        buffer + readByte,
        bufferSize - readByte,
        targetDesc.getValue().lineDirectiveMode);
    readByte += decodeBool(
        buffer + readByte,
        bufferSize - readByte,
        targetDesc.getValue().forceGLSLScalarBufferLayout);
    readByte += decodeUint32(
        buffer + readByte,
        bufferSize - readByte,
        targetDesc.getValue().compilerOptionEntryCount);

    if (targetDesc.getValue().compilerOptionEntryCount > 0)
    {
        slang::CompilerOptionEntry* entries = (slang::CompilerOptionEntry*)targetDesc.allocate(
            sizeof(slang::CompilerOptionEntry) * targetDesc.getValue().compilerOptionEntryCount);
        readByte += decodeStructArray(
            buffer + readByte,
            bufferSize - readByte,
            entries,
            targetDesc.getValue().compilerOptionEntryCount);
        targetDesc.getValue().compilerOptionEntries = entries;
    }

    return readByte;
}

size_t ParameterDecoder::decodeStruct(
    const uint8_t* buffer,
    int64_t bufferSize,
    ValueDecoder<slang::SpecializationArg>& specializationArg)
{
    SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

    size_t readByte = 0;
    readByte = decodeEnumValue(buffer, bufferSize, specializationArg.getValue().kind);

    // TODO: Special handle to address decode is needed.
    uint64_t address = 0;
    readByte += decodeAddress(buffer + readByte, bufferSize - readByte, address);
    (void)address;

    return readByte;
}
} // namespace SlangRecord
