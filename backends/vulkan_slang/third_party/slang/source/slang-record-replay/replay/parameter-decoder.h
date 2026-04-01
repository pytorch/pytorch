#ifndef PARAMETER_DECODER_H
#define PARAMETER_DECODER_H

#include "../util/record-format.h"
#include "../util/record-utility.h"
#include "decoder-helper.h"
#include "slang.h"

#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace SlangRecord
{
class ParameterDecoder
{
public:
    static size_t decodeInt8(const uint8_t* buffer, int64_t bufferSize, int8_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeUint8(const uint8_t* buffer, int64_t bufferSize, uint8_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeInt16(const uint8_t* buffer, int64_t bufferSize, int16_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeUint16(const uint8_t* buffer, int64_t bufferSize, uint16_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeInt32(const uint8_t* buffer, int64_t bufferSize, int32_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeUint32(const uint8_t* buffer, int64_t bufferSize, uint32_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeInt64(const uint8_t* buffer, int64_t bufferSize, int64_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeUint64(const uint8_t* buffer, int64_t bufferSize, uint64_t& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeFloat(const uint8_t* buffer, int64_t bufferSize, float& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeDouble(const uint8_t* buffer, int64_t bufferSize, double& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }
    static size_t decodeBool(const uint8_t* buffer, int64_t bufferSize, bool& value)
    {
        return decodeValue(buffer, bufferSize, value);
    }

    template<typename T>
    static size_t decodeEnumValue(const uint8_t* buffer, size_t bufferSize, T& value)
    {
        uint32_t decodedValue;
        size_t readByte = decodeValue(buffer, bufferSize, decodedValue);
        value = static_cast<T>(decodedValue);
        return readByte;
    }

    static size_t decodeString(
        const uint8_t* buffer,
        int64_t bufferSize,
        PointerDecoder<char*>& typeDecoder);

    static size_t decodePointer(
        const uint8_t* buffer,
        int64_t bufferSize,
        PointerDecoder<void*>& pointerDecoder);

    static size_t decodeAddress(
        const uint8_t* buffer,
        int64_t bufferSize,
        SlangRecord::AddressFormat& address)
    {
        return decodeValue(buffer, bufferSize, address);
    }
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<SlangGlobalSessionDesc>& sessionDesc);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::SessionDesc>& sessionDesc);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::PreprocessorMacroDesc>& desc);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::CompilerOptionEntry>& entry);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::CompilerOptionValue>& value);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::TargetDesc>& targetDesc);
    static size_t decodeStruct(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<slang::SpecializationArg>& specializationArg);

    template<typename T>
    static size_t decodeValueArray(
        const uint8_t* buffer,
        int64_t bufferSize,
        ValueDecoder<T>* valueArray,
        size_t count)
    {
        if (count == 0 && bufferSize == 0)
        {
            return 0;
        }

        size_t readByte = 0;
        for (size_t i = 0; i < count; ++i)
        {
            readByte += decodeValue(buffer + readByte, bufferSize - readByte, valueArray[i]);
        }
        return readByte;
    }

    static size_t decodeStringArray(
        const uint8_t* buffer,
        int64_t bufferSize,
        char** outputArray,
        size_t count)
    {
        if (count == 0 && bufferSize == 0)
        {
            return 0;
        }
        SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

        size_t readByte = 0;
        for (uint32_t i = 0; i < count; i++)
        {
            PointerDecoder<char*> item;
            readByte += decodeString(buffer + readByte, bufferSize - readByte, item);

            // Copy the search path
            outputArray[i] = item.getPointer();
        }
        return readByte;
    }

    template<typename T>
    static size_t decodeStructArray(
        const uint8_t* buffer,
        int64_t bufferSize,
        T* outputArray,
        size_t count)
    {
        if (count == 0 && bufferSize == 0)
        {
            return 0;
        }
        SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

        size_t bufferRead = 0;
        for (size_t i = 0; i < count; ++i)
        {
            ValueDecoder<T> item;
            bufferRead += decodeStruct(buffer + bufferRead, bufferSize - bufferRead, item);
            outputArray[i] = item.getValue();
        }
        return bufferRead;
    }

    static size_t decodeAddressArray(
        const uint8_t* buffer,
        int64_t bufferSize,
        uint64_t* addressArray,
        size_t count)
    {
        if (count == 0 && bufferSize == 0)
        {
            return 0;
        }
        SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

        size_t bufferRead = 0;
        for (size_t i = 0; i < count; ++i)
        {
            bufferRead +=
                decodeAddress(buffer + bufferRead, bufferSize - bufferRead, addressArray[i]);
        }

        return bufferRead;
    }


private:
    template<typename T>
    static size_t decodeValue(const uint8_t* buffer, int64_t bufferSize, T& value)
    {
        SLANG_RECORD_ASSERT((buffer != nullptr) && (bufferSize > 0));

        int64_t dataSize = sizeof(T);

        SLANG_RECORD_ASSERT(bufferSize >= dataSize);

        size_t bytesRead = 0;
        bytesRead = dataSize;
        memcpy(&value, buffer, dataSize);

        return bytesRead;
    }
};
} // namespace SlangRecord

#endif // PARAMETER_DECODER_H
