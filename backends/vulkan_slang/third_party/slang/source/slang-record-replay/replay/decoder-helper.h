#ifndef SLANG_DECODER_HELPER_H
#define SLANG_DECODER_HELPER_H

#include "../../core/slang-list.h"
#include "../util/record-format.h"
#include "slang-com-helper.h"
#include "slang.h"

#include <stdint.h>

namespace SlangRecord
{

// This class is used to allocate memory for the type decoder
class DecoderAllocatorSingleton
{
public:
    static DecoderAllocatorSingleton* getInstance();
    void* allocate(size_t size);
    ~DecoderAllocatorSingleton();

private:
    DecoderAllocatorSingleton() = default;
    Slang::List<void*> m_allocations;
};

class DecoderBase
{
public:
    virtual ~DecoderBase() = default;
    void* allocate(size_t size) { return m_allocator->allocate(size); }

protected:
    DecoderAllocatorSingleton* m_allocator = DecoderAllocatorSingleton::getInstance();
};

// We don't allow pointer type to be used as a template parameter
template<typename T, typename U = typename std::enable_if<!std::is_pointer<T>::value>::type>
class ValueDecoder : public DecoderBase
{
public:
    T& getValue() { return m_value; }

protected:
    T m_value{};
};

template<typename T, typename = typename std::enable_if<!std::is_pointer<T>::value>::type>
class StructDecoder : public ValueDecoder<T>
{
public:
    using Super = ValueDecoder<T>;
    size_t decode(const uint8_t* buffer, int64_t bufferSize);
};

// We only allow pointer type to be used as a template parameter
template<typename T, typename U = typename std::enable_if<std::is_pointer<T>::value>::type>
class PointerDecoder : public DecoderBase
{
public:
    void setPointer(void* data) { m_pointer = static_cast<T>(data); }
    T getPointer() const { return m_pointer; }
    void setPointerAddress(uint64_t address) { m_pointerAddress = address; }
    void setDataSize(size_t size) { m_dataSize = size; }
    size_t getDataSize() const { return m_dataSize; }

private:
    T m_pointer{nullptr};
    uint64_t m_pointerAddress = 0;
    size_t m_dataSize = 0;
};

class BlobDecoder
{
private:
    PointerDecoder<void*> m_blobData;

    class BlobImpl : public slang::IBlob
    {
    public:
        // ISlangUnknown
        virtual SLANG_NO_THROW SlangResult SLANG_MCALL
        queryInterface(SlangUUID const& uuid, void** outObject) override
        {
            if (uuid == ISlangUnknown::getTypeGuid() || uuid == ISlangBlob::getTypeGuid())
            {
                *outObject = static_cast<ISlangBlob*>(this);
                return SLANG_OK;
            }
            *outObject = nullptr;
            return SLANG_E_NO_INTERFACE;
        }

        virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return 1; }
        virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return 1; }

        BlobImpl(const PointerDecoder<void*>* blobData)
            : m_pBlobData(blobData)
        {
        }
        virtual SLANG_NO_THROW void const* SLANG_MCALL getBufferPointer() SLANG_OVERRIDE
        {
            return m_pBlobData->getPointer();
        }
        virtual SLANG_NO_THROW size_t SLANG_MCALL getBufferSize() SLANG_OVERRIDE
        {
            return m_pBlobData->getDataSize();
        }

    private:
        const PointerDecoder<void*>* m_pBlobData;
    };

    BlobImpl m_blobImpl{&m_blobData};
    AddressFormat m_address{0};

public:
    size_t decode(const uint8_t* buffer, int64_t bufferSize);
    slang::IBlob* getBlob() { return &m_blobImpl; }
};

using StringDecoder = PointerDecoder<char*>;
} // namespace SlangRecord

#endif // SLANG_RECORD_DECODER_HELPER_H
