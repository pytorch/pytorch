#include "slang-riff.h"

#include "slang-com-helper.h"
#include "slang-hex-dump-util.h"

namespace Slang
{

/* static */ int64_t RiffUtil::calcChunkTotalSize(const RiffHeader& chunk)
{
    size_t size = chunk.size + sizeof(RiffHeader);
    return getPadSize(size);
}

/* static */ SlangResult RiffUtil::skip(
    const RiffHeader& chunk,
    Stream* stream,
    int64_t* remainingBytesInOut)
{
    int64_t chunkSize = calcChunkTotalSize(chunk);
    if (remainingBytesInOut)
    {
        *remainingBytesInOut -= chunkSize;
    }

    // Skip the payload (we don't need to skip the Chunk because that was already read
    SLANG_RETURN_ON_FAIL(stream->seek(SeekOrigin::Current, chunkSize - sizeof(RiffHeader)));
    return SLANG_OK;
}

/* static */ SlangResult RiffUtil::readChunk(Stream* stream, RiffHeader& outChunk)
{
    size_t readBytes;
    SLANG_RETURN_ON_FAIL(stream->read(&outChunk, sizeof(RiffHeader), readBytes));
    // TODO(JS): Could handle endianness issues here...
    return (readBytes == sizeof(RiffHeader)) ? SLANG_OK : SLANG_FAIL;
}

/* static */ SlangResult RiffUtil::writeData(
    const RiffHeader* header,
    size_t headerSize,
    const void* payload,
    size_t payloadSize,
    Stream* out)
{
    SLANG_ASSERT(uint64_t(payloadSize) <= uint64_t(0xfffffffff));
    SLANG_ASSERT(headerSize >= sizeof(RiffHeader));

    // TODO(JS): Could handle endianness here

    RiffHeader chunk;
    chunk.type = header->type;
    chunk.size = uint32_t(headerSize - sizeof(RiffHeader) + payloadSize);

    // The chunk
    SLANG_RETURN_ON_FAIL(out->write(&chunk, sizeof(RiffHeader)));

    // Remainder of header
    if (headerSize > sizeof(RiffHeader))
    {
        // The rest of the header
        SLANG_RETURN_ON_FAIL(out->write(header + 1, headerSize - sizeof(RiffHeader)));
    }

    // Write the payload
    SLANG_RETURN_ON_FAIL(out->write(payload, payloadSize));

    // The riff spec requires all chunks are 4 byte aligned (even if size is not)
    size_t padSize = getPadSize(payloadSize);
    if (padSize - payloadSize)
    {
        uint8_t end[kRiffPadSize] = {0};
        SLANG_RETURN_ON_FAIL(out->write(end, padSize - payloadSize));
    }

    return SLANG_OK;
}

/* static */ SlangResult RiffUtil::readPayload(
    Stream* stream,
    size_t size,
    void* outData,
    size_t& outReadSize)
{
    outReadSize = 0;

    SLANG_RETURN_ON_FAIL(stream->readExactly(outData, size));

    const size_t alignedSize = getPadSize(size);
    // Skip to the alignment
    if (alignedSize > size)
    {
        SLANG_RETURN_ON_FAIL(stream->seek(SeekOrigin::Current, alignedSize - size));
    }
    outReadSize = alignedSize;
    return SLANG_OK;
}

/* static */ SlangResult RiffUtil::readData(
    Stream* stream,
    RiffHeader* outHeader,
    size_t headerSize,
    List<uint8_t>& data)
{
    RiffHeader chunk;
    SLANG_RETURN_ON_FAIL(readChunk(stream, chunk));
    if (chunk.size < headerSize)
    {
        return SLANG_FAIL;
    }

    *outHeader = chunk;

    // Read the header
    if (headerSize > sizeof(RiffHeader))
    {
        SLANG_RETURN_ON_FAIL(stream->readExactly(outHeader + 1, headerSize - sizeof(RiffHeader)));
    }

    const size_t payloadSize = chunk.size - (headerSize - sizeof(RiffHeader));
    size_t readSize;
    data.setCount(payloadSize);
    return readPayload(stream, payloadSize, data.getBuffer(), readSize);
}

/* static */ SlangResult RiffUtil::readHeader(Stream* stream, RiffListHeader& outHeader)
{
    // Need to read the chunk header
    SLANG_RETURN_ON_FAIL(readChunk(stream, outHeader.chunk));
    outHeader.subType = 0;

    if (isListType(outHeader.chunk.type))
    {
        // Read the sub type
        SLANG_RETURN_ON_FAIL(
            stream->readExactly(&outHeader.subType, sizeof(RiffListHeader) - sizeof(RiffHeader)));
    }

    return SLANG_OK;
}

namespace
{ // anonymous

struct DumpVisitor : public RiffContainer::Visitor
{
    typedef RiffContainer::Chunk Chunk;
    typedef RiffContainer::ListChunk ListChunk;
    typedef RiffContainer::DataChunk DataChunk;


    // Visitor
    virtual SlangResult enterList(ListChunk* list) SLANG_OVERRIDE
    {
        _dumpIndent();
        // If it's the root it's 'riff'
        _dumpRiffType(list == m_rootChunk ? RiffFourCC::kRiff : RiffFourCC::kList);
        m_writer.put(" ");
        _dumpRiffType(list->getSubType());
        m_writer.put("\n");
        m_indent++;
        return SLANG_OK;
    }
    virtual SlangResult handleData(DataChunk* data) SLANG_OVERRIDE
    {
        _dumpIndent();
        // Write out the name
        _dumpRiffType(data->m_fourCC);
        m_writer.put(" ");

        const RiffHashCode hash = data->calcHash();

        // We don't know in general what the contents is or means... but we can display a hash
        HexDumpUtil::dump(uint32_t(hash), m_writer.getWriter());
        m_writer.put(" ");

        m_writer.put("\n");
        return SLANG_OK;
    }
    virtual SlangResult leaveList(ListChunk* list) SLANG_OVERRIDE
    {
        SLANG_UNUSED(list);
        m_indent--;
        return SLANG_OK;
    }

    DumpVisitor(WriterHelper writer, Chunk* rootChunk)
        : m_writer(writer), m_indent(0), m_rootChunk(rootChunk)
    {
    }

    void _dumpIndent()
    {
        for (int i = 0; i < m_indent; ++i)
        {
            m_writer.put("  ");
        }
    }
    void _dumpRiffType(FourCC fourCC)
    {
        char c[5];
        for (int i = 0; i < 4; ++i)
        {
            c[i] = char(fourCC);
            fourCC >>= 8;
        }
        c[4] = 0;
        m_writer.put(c);
    }

    Chunk* m_rootChunk;

    int m_indent;
    WriterHelper m_writer;
};

} // namespace

/* static */ void RiffUtil::dump(RiffContainer::Chunk* chunk, WriterHelper writer)
{
    DumpVisitor visitor(writer, chunk);
    chunk->visit(&visitor);
}

/* static */ SlangResult RiffUtil::write(
    RiffContainer::ListChunk* list,
    bool isRoot,
    Stream* stream)
{
    RiffListHeader listHeader;

    listHeader.chunk.type = isRoot ? RiffFourCC::kRiff : RiffFourCC::kList;
    listHeader.chunk.size = uint32_t(list->m_payloadSize);
    listHeader.subType = list->getSubType();

    // Write the header
    SLANG_RETURN_ON_FAIL(stream->write(&listHeader, sizeof(listHeader)));

    // Write the contained chunks
    Chunk* chunk = list->m_containedChunks;
    while (chunk)
    {
        switch (chunk->m_kind)
        {
        case Chunk::Kind::List:
            {
                auto listChunk = static_cast<ListChunk*>(chunk);
                // It's a container
                SLANG_RETURN_ON_FAIL(write(listChunk, false, stream));
                break;
            }
        case Chunk::Kind::Data:
            {
                auto dataChunk = static_cast<DataChunk*>(chunk);

                // Must be a regular chunk with data
                RiffHeader chunkHeader;
                chunkHeader.type = dataChunk->m_fourCC;
                chunkHeader.size = uint32_t(dataChunk->m_payloadSize);

                SLANG_RETURN_ON_FAIL(stream->write(&chunkHeader, sizeof(chunkHeader)));

                RiffContainer::Data* data = dataChunk->m_dataList;
                while (data)
                {
                    SLANG_RETURN_ON_FAIL(stream->write(data->getPayload(), data->getSize()));

                    // Next but of data
                    data = data->m_next;
                }

                // Need to write for alignment
                const size_t remainingSize =
                    getPadSize(dataChunk->m_payloadSize) - dataChunk->m_payloadSize;

                if (remainingSize)
                {
                    static const uint8_t trailing[kRiffPadSize] = {0};
                    SLANG_RETURN_ON_FAIL(stream->write(trailing, remainingSize));
                }
            }
        default:
            break;
        }

        // Next
        chunk = chunk->m_next;
    }

    return SLANG_OK;
}

/* static */ SlangResult RiffUtil::write(RiffContainer* container, Stream* stream)
{
    return write(container->getRoot(), true, stream);
}

/* static */ SlangResult RiffUtil::read(Stream* stream, RiffContainer& outContainer)
{
    typedef RiffContainer::ScopeChunk ScopeChunk;
    outContainer.reset();

    size_t remaining;
    {
        RiffListHeader header;

        SLANG_RETURN_ON_FAIL(readHeader(stream, header));
        if (!isListType(header.chunk.type))
        {
            return SLANG_FAIL;
        }

        remaining = getPadSize(header.chunk.size) - (sizeof(RiffListHeader) - sizeof(RiffHeader));
        outContainer.startChunk(Chunk::Kind::List, header.subType);
    }

    List<size_t> remainingStack;
    while (true)
    {
        // It must be the end
        if (remaining == 0)
        {
            // If it's a container then we pop container
            outContainer.endChunk();
            if (remainingStack.getCount() <= 0)
            {
                break;
            }

            remaining = remainingStack.getLast();
            remainingStack.removeLast();
        }
        else
        {
            RiffListHeader header;
            SLANG_RETURN_ON_FAIL(readHeader(stream, header));

            // The amount of data can't be larger than what remains
            if (header.chunk.size > remaining)
            {
                return SLANG_FAIL;
            }

            if (header.chunk.type == RiffFourCC::kList)
            {
                if (header.chunk.size & kRiffPadMask)
                {
                    SLANG_ASSERT(!"A list chunk can only have divisible by 2 size");
                    return SLANG_FAIL;
                }

                // Work out the pad size
                const size_t padSize = getPadSize(header.chunk.size);

                // Subtract the size of this chunk from remaining of the current chunk
                remaining -= sizeof(RiffHeader) + padSize;
                // Push it, for when we hit the end
                remainingStack.add(remaining);

                // Work out how much remains in this container
                remaining = padSize - (sizeof(RiffListHeader) - sizeof(RiffHeader));

                // Start a container
                outContainer.startChunk(Chunk::Kind::List, header.subType);
            }
            else
            {
                ScopeChunk scopeChunk(&outContainer, Chunk::Kind::Data, header.chunk.type);
                RiffContainer::Data* data = outContainer.addData();

                outContainer.setPayload(data, nullptr, header.chunk.size);

                size_t readSize;
                SLANG_RETURN_ON_FAIL(
                    readPayload(stream, header.chunk.size, data->getPayload(), readSize));

                // All read sizes must end up aligned
                SLANG_ASSERT((readSize & kRiffPadMask) == 0);

                // Correct remaining
                remaining -= sizeof(RiffHeader) + readSize;
            }
        }
    }

    return outContainer.isFullyConstructed() ? SLANG_OK : SLANG_FAIL;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RiffContainer::Chunk !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult RiffContainer::Chunk::visit(Visitor* visitor)
{
    switch (m_kind)
    {
    case Kind::Data:
        {
            return visitor->handleData(static_cast<DataChunk*>(this));
        }
    case Kind::List:
        {
            auto list = static_cast<ListChunk*>(this);
            SLANG_RETURN_ON_FAIL(visitor->enterList(list));

            Chunk* chunk = list->m_containedChunks;
            while (chunk)
            {
                SLANG_RETURN_ON_FAIL(chunk->visit(visitor));

                chunk = chunk->m_next;
            }

            SLANG_RETURN_ON_FAIL(visitor->leaveList(list));
            return SLANG_OK;
        }
    default:
        return SLANG_FAIL;
    }
}

SlangResult RiffContainer::Chunk::visitPreOrder(VisitorCallback callback, void* data)
{
    switch (m_kind)
    {
    case Kind::Data:
        {
            return callback(this, data);
        }
    case Kind::List:
        {
            auto list = static_cast<ListChunk*>(this);
            // Do this containing node first
            SLANG_RETURN_ON_FAIL(callback(this, data));

            // Do the contents next
            Chunk* chunk = list->m_containedChunks;
            while (chunk)
            {
                SLANG_RETURN_ON_FAIL(chunk->visitPreOrder(callback, data));
                chunk = chunk->m_next;
            }
            return SLANG_OK;
        }
    default:
        return SLANG_FAIL;
    }
}

SlangResult RiffContainer::Chunk::visitPostOrder(VisitorCallback callback, void* data)
{
    switch (m_kind)
    {
    case Kind::Data:
        {
            return callback(this, data);
        }
    case Kind::List:
        {
            auto list = static_cast<ListChunk*>(this);

            // Do the contents first
            Chunk* chunk = list->m_containedChunks;
            while (chunk)
            {
                SLANG_RETURN_ON_FAIL(chunk->visitPostOrder(callback, data));
                chunk = chunk->m_next;
            }
            // Then the list node (so a post order)
            SLANG_RETURN_ON_FAIL(callback(this, data));
            return SLANG_OK;
        }
    default:
        return SLANG_FAIL;
    }
}

size_t RiffContainer::Chunk::calcPayloadSize()
{
    switch (m_kind)
    {
    case Kind::Data:
        return static_cast<DataChunk*>(this)->calcPayloadSize();
    case Kind::List:
        return static_cast<ListChunk*>(this)->calcPayloadSize();
    default:
        return 0;
    }
}

RiffContainer::Data* RiffContainer::Chunk::getSingleData() const
{
    return (m_kind == Kind::Data) ? static_cast<const DataChunk*>(this)->getSingleData() : nullptr;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!! RiffContainer::ListChunk !!!!!!!!!!!!!!!!!!!!!!

size_t RiffContainer::ListChunk::calcPayloadSize()
{
    // Have to include the part of the header not taken up by the RiffHeader
    size_t size = sizeof(RiffListHeader) - sizeof(RiffHeader);
    Chunk* chunk = m_containedChunks;
    while (chunk)
    {
        size_t chunkSize = chunk->m_payloadSize + sizeof(RiffHeader);
        // Align the contained chunk size
        size += RiffUtil::getPadSize(chunkSize);

        chunk = chunk->m_next;
    }
    return size;
}

RiffContainer::Chunk* RiffContainer::ListChunk::findContained(FourCC fourCC) const
{
    Chunk* chunk = m_containedChunks;
    while (chunk)
    {
        if (chunk->m_fourCC == fourCC)
        {
            return chunk;
        }
        chunk = chunk->m_next;
    }
    return nullptr;
}

void RiffContainer::ListChunk::findContained(FourCC type, List<ListChunk*>& out)
{
    Chunk* chunk = m_containedChunks;
    while (chunk)
    {
        if (chunk->m_fourCC == type && chunk->m_kind == Chunk::Kind::List)
        {
            out.add(static_cast<ListChunk*>(chunk));
        }
        chunk = chunk->m_next;
    }
}

void RiffContainer::ListChunk::findContained(FourCC type, List<DataChunk*>& out)
{
    Chunk* chunk = m_containedChunks;
    while (chunk)
    {
        if (chunk->m_fourCC == type && chunk->m_kind == Chunk::Kind::Data)
        {
            out.add(static_cast<DataChunk*>(chunk));
        }
        chunk = chunk->m_next;
    }
}

RiffContainer::ListChunk* RiffContainer::ListChunk::findContainedList(FourCC type)
{
    Chunk* chunk = m_containedChunks;
    while (chunk)
    {
        if (chunk->m_fourCC == type && chunk->m_kind == Chunk::Kind::List)
        {
            return static_cast<ListChunk*>(chunk);
        }
        chunk = chunk->m_next;
    }
    return nullptr;
}

RiffContainer::Data* RiffContainer::ListChunk::findContainedData(FourCC type) const
{
    Chunk* found = findContained(type);
    if (found && found->m_kind == Kind::Data)
    {
        DataChunk* dataChunk = static_cast<DataChunk*>(found);
        // Assumes that there is a single data chunk

        Data* data = dataChunk->m_dataList;
        if (data && data->m_next == nullptr)
        {
            return data;
        }
    }
    return nullptr;
}

void* RiffContainer::ListChunk::findContainedData(FourCC type, size_t minSize) const
{
    Data* data = findContainedData(type);
    return (data && data->m_size >= minSize) ? data->getPayload() : nullptr;
}

static RiffContainer::ListChunk* _findListRec(RiffContainer::ListChunk* list, FourCC subType)
{
    RiffContainer::Chunk* chunk = list->m_containedChunks;
    while (chunk)
    {
        if (auto childList = as<RiffContainer::ListChunk>(chunk))
        {
            // Test if the child is the subtype, if so we are done
            if (childList->getSubType() == subType)
            {
                return childList;
            }
            auto found = _findListRec(childList, subType);
            if (found)
            {
                return found;
            }
        }
        chunk = chunk->m_next;
    }
    return nullptr;
}

/* static */ RiffContainer::ListChunk* RiffContainer::ListChunk::findListRec(FourCC subType)
{
    return (getSubType() == subType) ? this : _findListRec(this, subType);
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!! RiffContainer::DataChunk !!!!!!!!!!!!!!!!!!!!!!

RiffContainer::Data* RiffContainer::DataChunk::getSingleData() const
{
    Data* data = m_dataList;
    return (data && data->m_next == nullptr) ? data : nullptr;
}

RiffReadHelper RiffContainer::DataChunk::asReadHelper() const
{
    Data* data = getSingleData();
    if (data)
    {
        return RiffReadHelper((const uint8_t*)data->getPayload(), data->getSize());
    }
    return RiffReadHelper(nullptr, 0);
}

RiffHashCode RiffContainer::DataChunk::calcHash() const
{
    RiffHashCode hash = 0;

    Data* data = m_dataList;
    while (data)
    {
        // This is a little contrived (in that we don't use the function getHashCode), but the
        // reason to be careful is we want the same result however many Data blocks there are.
        const char* buffer = (const char*)data->getPayload();
        const size_t size = data->getSize();

        for (size_t i = 0; i < size; ++i)
        {
            hash = RiffHashCode(buffer[i]) + (hash << 6) + (hash << 16) - hash;
        }

        data = data->m_next;
    }

    return hash;
}

size_t RiffContainer::DataChunk::calcPayloadSize() const
{
    size_t size = 0;
    Data* data = m_dataList;
    while (data)
    {
        size += data->getSize();
        data = data->m_next;
    }
    return size;
}

void RiffContainer::DataChunk::getPayload(void* inDst) const
{
    uint8_t* dst = (uint8_t*)inDst;

    Data* data = m_dataList;
    while (data)
    {
        const size_t size = data->getSize();
        ::memcpy(dst, data->getPayload(), size);

        dst += size;
        data = data->m_next;
    }
}

bool RiffContainer::DataChunk::isEqual(const void* inData, size_t count) const
{
    const uint8_t* src = (const uint8_t*)inData;

    Data* data = m_dataList;
    while (data)
    {
        const size_t size = data->getSize();
        // Can't have more content than remaining
        // Contents must match
        if (size > count || ::memcmp(src, data->getPayload(), size) != 0)
        {
            return false;
        }

        src += size;
        count -= size;

        // Next data block
        data = data->m_next;
    }

    // If match must be at the end
    return count == 0;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! RiffContainer !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RiffContainer::RiffContainer()
    : m_arena(4096)
{
    m_rootList = nullptr;
    m_listChunk = nullptr;
    m_dataChunk = nullptr;
}

void RiffContainer::reset()
{
    m_arena.reset();

    m_rootList = nullptr;
    m_listChunk = nullptr;
    m_dataChunk = nullptr;
}

RiffContainer::ListChunk* RiffContainer::_newListChunk(FourCC subType)
{
    SLANG_ASSERT(!RiffUtil::isListType(subType));

    ListChunk* chunk = (ListChunk*)m_arena.allocate(sizeof(ListChunk));
    chunk->init(subType);
    return chunk;
}

RiffContainer::DataChunk* RiffContainer::_newDataChunk(FourCC type)
{
    SLANG_ASSERT(!RiffUtil::isListType(type));

    DataChunk* chunk = (DataChunk*)m_arena.allocate(sizeof(DataChunk));
    chunk->init(type);
    return chunk;
}

void RiffContainer::_addChunk(Chunk* chunk)
{
    if (m_listChunk)
    {
        chunk->m_parent = m_listChunk;
        Chunk*& next = m_listChunk->m_endChunk ? m_listChunk->m_endChunk->m_next
                                               : m_listChunk->m_containedChunks;
        SLANG_ASSERT(next == nullptr);
        next = chunk;
        m_listChunk->m_endChunk = chunk;
    }
}

void RiffContainer::setCurrentChunk(Chunk* chunk)
{
    SLANG_ASSERT(chunk);

    switch (chunk->m_kind)
    {
    case Chunk::Kind::Data:
        m_listChunk = nullptr;
        m_dataChunk = static_cast<RiffContainer::DataChunk*>(chunk);
        break;

    case Chunk::Kind::List:
        m_dataChunk = nullptr;
        m_listChunk = static_cast<RiffContainer::ListChunk*>(chunk);
        break;
    }
}

void RiffContainer::startChunk(Chunk::Kind kind, FourCC fourCC)
{
    SLANG_ASSERT(m_listChunk || m_rootList == nullptr);

    switch (kind)
    {
    case Chunk::Kind::Data:
        {
            // We can only start a data chunk if we are in a container, and we can't already be in
            // data chunk
            SLANG_ASSERT(m_listChunk && m_dataChunk == nullptr);

            DataChunk* chunk = _newDataChunk(fourCC);
            _addChunk(chunk);
            m_dataChunk = chunk;
            break;
        }
    case Chunk::Kind::List:
        {
            // We can't be in a data chunk
            SLANG_ASSERT(m_dataChunk == nullptr);

            ListChunk* list = _newListChunk(fourCC);

            // If this is the first, make it the root
            if (!m_rootList)
            {
                m_rootList = list;
            }

            _addChunk(list);

            m_listChunk = list;
            break;
        }
    }
}

void RiffContainer::endChunk()
{
    size_t chunkPayloadSize;

    // The chunk we are popping
    // Only keep track of this in debug builds
    [[maybe_unused]] Chunk* chunk = nullptr;

    ListChunk* parent;
    if (m_dataChunk)
    {
        chunk = m_dataChunk;

        parent = m_dataChunk->m_parent;
        chunkPayloadSize = m_dataChunk->m_payloadSize;

        m_dataChunk = nullptr;
    }
    else
    {
        chunk = m_listChunk;

        SLANG_ASSERT(m_listChunk && m_dataChunk == nullptr);
        parent = m_listChunk->m_parent;
        chunkPayloadSize = m_listChunk->m_payloadSize;
    }

    m_listChunk = parent;

    if (parent)
    {
        // Fix the size taking into account padding bytes requirement
        chunkPayloadSize = RiffUtil::getPadSize(chunkPayloadSize);
        // Update the parents size
        parent->m_payloadSize += sizeof(RiffHeader) + chunkPayloadSize;
    }

    // Check it's size seems ok
    SLANG_ASSERT(isChunkOk(chunk));
}

void RiffContainer::addDataChunk(FourCC dataFourCC, const void* data, size_t dataSizeInBytes)
{
    startChunk(Chunk::Kind::Data, dataFourCC);
    write(data, dataSizeInBytes);
    endChunk();
}

void RiffContainer::setPayload(Data* data, const void* payload, size_t size)
{
    // We must be in a data chunk
    SLANG_ASSERT(m_dataChunk);
    // The data shouldn't be set up
    SLANG_ASSERT(data->m_ownership == Ownership::Uninitialized);

    // Add current chunks data
    m_dataChunk->m_payloadSize += size;

    data->m_ownership = Ownership::Arena;
    data->m_size = size;

    if (size)
    {
        data->m_payload = m_arena.allocateAligned(size, kPayloadMinAlignment);
    }

    if (payload)
    {
        ::memcpy(data->m_payload, payload, size);
    }
}

void RiffContainer::moveOwned(Data* data, void* payload, size_t size)
{
    // We must be in a data chunk
    SLANG_ASSERT(m_dataChunk);
    // The data shouldn't be set up
    SLANG_ASSERT(data->m_ownership == Ownership::Uninitialized);

    // Add current chunks data
    m_dataChunk->m_payloadSize += size;

    data->m_ownership = Ownership::Owned;
    data->m_size = size;

    // The area will manage this block
    m_arena.addExternalBlock(payload, size);
    data->m_payload = payload;
}

void RiffContainer::setUnowned(Data* data, void* payload, size_t size)
{
    // We must be in a data chunk
    SLANG_ASSERT(m_dataChunk);
    // The data shouldn't be set up
    SLANG_ASSERT(data->m_ownership == Ownership::Uninitialized);
    // Add current chunks data
    m_dataChunk->m_payloadSize += size;

    data->m_ownership = Ownership::NotOwned;
    data->m_size = size;
    data->m_payload = payload;
}

RiffContainer::Data* RiffContainer::addData()
{
    // We must be in a chunk
    SLANG_ASSERT(m_dataChunk);

    Data* data = (Data*)m_arena.allocate(sizeof(Data));
    data->init();

    Data*& next = m_dataChunk->m_endData ? m_dataChunk->m_endData->m_next : m_dataChunk->m_dataList;
    SLANG_ASSERT(next == nullptr);

    // Add to linked list
    next = data;
    // Make this the new end
    m_dataChunk->m_endData = data;
    return data;
}

RiffContainer::Data* RiffContainer::makeSingleData(DataChunk* dataChunk)
{
    // There is no data
    if (dataChunk->m_dataList == nullptr)
    {
        return nullptr;
    }

    if (dataChunk->m_dataList->m_next == nullptr)
    {
        return dataChunk->m_dataList;
    }

    {
        Data* data = dataChunk->m_dataList;

        // Okay lets combine all into one block
        const size_t payloadSize = dataChunk->calcPayloadSize();

        void* dst = m_arena.allocateAligned(payloadSize, kPayloadMinAlignment);
        dataChunk->getPayload(dst);

        // Remove other datas
        data->m_next = nullptr;
        // Make this the end
        dataChunk->m_endData = data;

        // Point to the block with all of the data
        data->m_ownership = Ownership::Arena;
        data->m_payload = dst;
        data->m_size = payloadSize;

        return data;
    }
}

void RiffContainer::write(const void* inData, size_t size)
{
    // We must be in a chunk
    SLANG_ASSERT(m_dataChunk);
    // Get the last data chunk
    Data* endData = m_dataChunk->m_endData;
    if (endData)
    {
        uint8_t* end = ((uint8_t*)endData->m_payload) + endData->m_size;
        // See if can just add to end of current data
        if (end == m_arena.getCursor() && m_arena.allocateCurrentUnaligned(size))
        {
            ::memcpy(end, inData, size);
            endData->m_size += size;

            // Add current chunks data
            m_dataChunk->m_payloadSize += size;
            return;
        }
    }

    auto data = addData();
    setPayload(data, inData, size);
}

static SlangResult _isChunkOk(RiffContainer::Chunk* chunk, void* data)
{
    SLANG_UNUSED(data);
    return chunk->calcPayloadSize() == chunk->m_payloadSize ? SLANG_OK : SLANG_FAIL;
}

/* static */ bool RiffContainer::isChunkOk(Chunk* chunk)
{
    return SLANG_SUCCEEDED(chunk->visitPostOrder(&_isChunkOk, nullptr));
}

static SlangResult _calcAndSetSize(RiffContainer::Chunk* chunk, void* data)
{
    SLANG_UNUSED(data);
    chunk->m_payloadSize = chunk->calcPayloadSize();
    return SLANG_OK;
}

/* static */ void RiffContainer::calcAndSetSize(Chunk* chunk)
{
    chunk->visitPostOrder(&_calcAndSetSize, nullptr);
}


} // namespace Slang
