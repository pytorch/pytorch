#ifndef PARAMETER_ENCODER_H
#define PARAMETER_ENCODER_H

#include "../util/record-format.h"
#include "output-stream.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace SlangRecord
{
class ParameterRecorder
{
public:
    ParameterRecorder(OutputStream* stream)
        : m_stream(stream){};
    void recordInt8(int8_t value) { recordValue(value); }
    void recordUint8(uint8_t value) { recordValue(value); }
    void recordInt16(int16_t value) { recordValue(value); }
    void recordUint16(uint16_t value) { recordValue(value); }
    void recordInt32(int32_t value) { recordValue(value); }
    void recordUint32(uint32_t value) { recordValue(value); }
    void recordInt64(int64_t value) { recordValue(value); }
    void recordUint64(uint64_t value) { recordValue(value); }
    void recordFloat(float value) { recordValue(value); }
    void recordDouble(double value) { recordValue(value); }
    void recordBool(bool value) { recordValue(value); }

    template<typename T>
    void recordEnumValue(T value)
    {
        recordValue(static_cast<uint32_t>(value));
    }

    void recordString(const char* value);
    void recordPointer(const void* value, bool omitData = false, size_t size = 0);
    void recordPointer(ISlangBlob* blob);
    void recordAddress(const void* value)
    {
        recordValue(reinterpret_cast<SlangRecord::AddressFormat>(value));
    }
    void recordStruct(SlangGlobalSessionDesc const& desc);
    void recordStruct(slang::SessionDesc const& desc);
    void recordStruct(slang::PreprocessorMacroDesc const& desc);
    void recordStruct(slang::CompilerOptionEntry const& entry);
    void recordStruct(slang::CompilerOptionValue const& value);
    void recordStruct(slang::TargetDesc const& targetDesc);
    void recordStruct(slang::SpecializationArg const& specializationArg);

    template<typename T>
    void recordValueArray(const T* array, size_t count)
    {
        recordUint32((uint32_t)count);
        for (size_t i = 0; i < count; ++i)
        {
            recordValue(array[i]);
        }
    }

    void recordStringArray(const char* const* array, size_t count)
    {
        recordUint32((uint32_t)count);
        for (size_t i = 0; i < count; ++i)
        {
            recordString(array[i]);
        }
    }

    template<typename T>
    void recordStructArray(T const* array, size_t count)
    {
        recordUint32((uint32_t)count);
        for (size_t i = 0; i < count; ++i)
        {
            recordStruct(array[i]);
        }
    }

    template<typename T>
    void recordAddressArray(T* const* array, size_t count)
    {
        recordUint32((uint32_t)count);
        for (size_t i = 0; i < count; ++i)
        {
            recordAddress(array[i]);
        }
    }


private:
    template<typename T>
    void recordValue(T value)
    {
        m_stream->write(&value, sizeof(T));
    }
    OutputStream* m_stream;
};
} // namespace SlangRecord

#endif // PARAMETER_ENCODER_H
