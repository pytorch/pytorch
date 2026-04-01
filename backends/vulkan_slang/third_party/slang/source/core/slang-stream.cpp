#include "slang-stream.h"
#ifdef _WIN32
#include <share.h>
#endif
#include "slang-io.h"
#include "slang-process.h"

#include <stdio.h>
#include <thread>

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FileStream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult Stream::readExactly(void* buffer, size_t length)
{
    size_t readBytes;
    SLANG_RETURN_ON_FAIL(read(buffer, length, readBytes));
    return (readBytes == length) ? SLANG_OK : SLANG_FAIL;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FileStream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

FileStream::FileStream()
    : m_handle(nullptr), m_fileAccess(FileAccess::None), m_endReached(false)
{
}

SlangResult FileStream::init(const String& fileName, FileMode fileMode)
{
    const FileAccess access = (fileMode == FileMode::Open) ? FileAccess::Read : FileAccess::Write;
    return _init(fileName, fileMode, access, FileShare::None);
}

SlangResult FileStream::init(
    const String& fileName,
    FileMode fileMode,
    FileAccess access,
    FileShare share)
{
    return _init(fileName, fileMode, access, share);
}

SlangResult FileStream::_init(
    const String& fileName,
    FileMode fileMode,
    FileAccess access,
    // Only used on Windows
    [[maybe_unused]] FileShare share)
{
    // Make sure it's closed to start with
    close();

    if (access == FileAccess::None)
    {
        SLANG_ASSERT(!"FileAccess::None not valid to create a FileStream.");
        return SLANG_E_INVALID_ARG;
    }

    const char* mode = "rt";
    switch (fileMode)
    {
    case FileMode::Create:
        if (access == FileAccess::Read)
        {
            SLANG_ASSERT(!"Read-only access is incompatible with Create mode.");
            return SLANG_E_INVALID_ARG;
        }
        else if (access == FileAccess::ReadWrite)
        {
            mode = "w+b";
        }
        else
        {
            mode = "wb";
        }
        break;
    case FileMode::Open:
        if (access == FileAccess::Read)
        {
            mode = "rb";
        }
        else if (access == FileAccess::ReadWrite)
        {
            mode = "r+b";
        }
        else
        {
            mode = "wb";
        }
        break;
    case FileMode::CreateNew:
        if (File::exists(fileName))
        {
            return SLANG_E_CANNOT_OPEN;
        }
        if (access == FileAccess::Read)
        {
            SLANG_ASSERT(!"Read-only access is incompatible with Create mode.");
            return SLANG_E_INVALID_ARG;
        }
        else if (access == FileAccess::ReadWrite)
        {
            mode = "w+b";
        }
        else
        {
            mode = "wb";
        }
        break;
    case FileMode::Append:
        if (access == FileAccess::Read)
        {
            SLANG_ASSERT(!"Read-only access is incompatible with Append mode.");
            return SLANG_E_INVALID_ARG;
        }
        else if (access == FileAccess::ReadWrite)
        {
            mode = "a+b";
        }
        else
        {
            mode = "ab";
        }
        break;
    default:
        break;
    }
#ifdef _WIN32

    // NOTE! This works because we know all the characters in the mode
    // are encoded directly as the same value in a wchar_t.
    //
    // Work out the length *including* terminating 0
    const Index modeLength = Index(::strlen(mode)) + 1;
    wchar_t wideMode[8];
    SLANG_ASSERT(modeLength <= SLANG_COUNT_OF(wideMode));

    // Copy to wchar_t
    for (Index i = 0; i < modeLength; ++i)
    {
        wideMode[i] = wchar_t(mode[i]);
    }

    int shFlag = _SH_DENYRW;
    switch (share)
    {
    case FileShare::None:
        shFlag = _SH_DENYRW;
        break;
    case FileShare::ReadOnly:
        shFlag = _SH_DENYWR;
        break;
    case FileShare::WriteOnly:
        shFlag = _SH_DENYRD;
        break;
    case FileShare::ReadWrite:
        shFlag = _SH_DENYNO;
        break;
    default:
        SLANG_ASSERT(!"Invalid file share mode.");
        return SLANG_FAIL;
    }

    if (share == FileShare::None)
    {
        m_handle = _wfsopen(fileName.toWString(), wideMode, _SH_DENYNO);
    }
    else
    {
        m_handle = _wfsopen(fileName.toWString(), wideMode, shFlag);
    }
#else
    m_handle = fopen(fileName.getBuffer(), mode);
#endif
    if (!m_handle)
    {
        return SLANG_E_CANNOT_OPEN;
    }

    // Just set the access specified
    m_fileAccess = access;
    return SLANG_OK;
}

FileStream::~FileStream()
{
    close();
}

Int64 FileStream::getPosition()
{
#if defined(_WIN32) || defined(__CYGWIN__)
    fpos_t pos;
    fgetpos(m_handle, &pos);
    return pos;
#elif defined(__APPLE__)
    return ftell(m_handle);
#else
    fpos64_t pos;
    fgetpos64(m_handle, &pos);
    return *(Int64*)(&pos);
#endif
}

SlangResult FileStream::seek(SeekOrigin seekOrigin, Int64 offset)
{
    int fseekOrigin;
    switch (seekOrigin)
    {
    case SeekOrigin::Start:
        fseekOrigin = SEEK_SET;
        break;
    case SeekOrigin::End:
        fseekOrigin = SEEK_END;
        break;
    case SeekOrigin::Current:
        fseekOrigin = SEEK_CUR;
        break;
    default:
        SLANG_ASSERT(!"Unsupported seek origin.");
        return SLANG_FAIL;
    }

    // If endReached is intended to be like feof - then doing a seek will reset it
    m_endReached = false;

#ifdef _WIN32
    int rs = _fseeki64(m_handle, offset, fseekOrigin);
#else
    int rs = fseek(m_handle, (long int)offset, fseekOrigin);
#endif

    // If rs != 0 then the the seek failed
    SLANG_ASSERT(rs == 0);

    return (rs == 0) ? SLANG_OK : SLANG_FAIL;
}

SlangResult FileStream::read(void* buffer, size_t length, size_t& outBytesRead)
{
    auto bytesRead = fread_s(buffer, length, 1, length, m_handle);

    outBytesRead = bytesRead;
    if (bytesRead == 0 && length > 0)
    {
        // If we have reached the end, then reading nothing is ok.
        if (!m_endReached)
        {
            // If we are not at the end of the file we should be able to read some bytes
            if (!feof(m_handle))
            {
                return SLANG_FAIL;
            }
            m_endReached = true;
        }
    }
    return SLANG_OK;
}

SlangResult FileStream::write(const void* buffer, size_t length)
{
    auto bytesWritten = fwrite(buffer, 1, length, m_handle);
    return (bytesWritten == length) ? SLANG_OK : SLANG_FAIL;
}

SlangResult FileStream::flush()
{
    if (m_handle && canWrite())
    {
        fflush(m_handle);
        return SLANG_OK;
    }
    return SLANG_E_NOT_AVAILABLE;
}

bool FileStream::canRead()
{
    return ((int)m_fileAccess & (int)FileAccess::Read) != 0;
}

bool FileStream::canWrite()
{
    return ((int)m_fileAccess & (int)FileAccess::Write) != 0;
}

void FileStream::close()
{
    if (m_handle)
    {
        fclose(m_handle);
        m_handle = nullptr;

        // If closed, can neither read or write
        m_fileAccess = FileAccess::None;
    }
}

bool FileStream::isEnd()
{
    return m_endReached;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! MemoryStreamBase !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult MemoryStreamBase::seek(SeekOrigin origin, Int64 offset)
{
    Int64 pos = 0;
    switch (origin)
    {
    case SeekOrigin::Start:
        pos = offset;
        break;
    case SeekOrigin::End:
        pos = Int64(m_contentsSize) + offset;
        break;
    case SeekOrigin::Current:
        pos = Int64(m_position) + offset;
        break;
    default:
        SLANG_ASSERT(!"Unsupported seek origin.");
        return SLANG_E_NOT_IMPLEMENTED;
    }

    m_atEnd = false;

    // Clamp to the valid range
    pos = (pos < 0) ? 0 : pos;
    pos = (pos > Int64(m_contentsSize)) ? Int64(m_contentsSize) : pos;

    m_position = ptrdiff_t(pos);
    return SLANG_OK;
}

SlangResult MemoryStreamBase::read(void* buffer, size_t length, size_t& outReadBytes)
{
    outReadBytes = 0;
    if (!canRead())
    {
        SLANG_ASSERT(!"Cannot read this stream.");
        return SLANG_FAIL;
    }

    const size_t maxRead = size_t(m_contentsSize - m_position);
    if (maxRead == 0 && length > 0)
    {
        // At end of stream
        m_atEnd = true;
        return SLANG_OK;
    }

    length = length > maxRead ? maxRead : length;

    ::memcpy(buffer, m_contents + m_position, length);
    m_position += ptrdiff_t(length);
    outReadBytes = length;

    return SLANG_OK;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! OwnedMemoryStream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult OwnedMemoryStream::write(const void* buffer, size_t length)
{
    if (!canWrite())
    {
        SLANG_ASSERT(!"Cannot write this stream.");
        return SLANG_FAIL;
    }

    if (m_position == m_ownedContents.getCount())
    {
        m_ownedContents.addRange((const uint8_t*)buffer, Index(length));
    }
    else
    {
        m_ownedContents.insertRange(m_position, (const uint8_t*)buffer, Index(length));
    }

    m_contents = m_ownedContents.getBuffer();
    m_contentsSize = ptrdiff_t(m_ownedContents.getCount());

    m_atEnd = false;

    m_position += ptrdiff_t(length);
    return SLANG_OK;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! BufferedReadStream !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

void BufferedReadStream::consume(Index byteCount)
{
    SLANG_ASSERT(Index(getCount()) >= byteCount && byteCount >= 0);
    m_startIndex += byteCount;
    if (getCount() == 0)
    {
        _resetBuffer();
    }
}

Int64 BufferedReadStream::getPosition()
{
    return m_stream ? (m_stream->getPosition() - getCount()) : 0;
}

SlangResult BufferedReadStream::seek(SeekOrigin origin, Int64 offset)
{
    if (!m_stream)
    {
        return SLANG_FAIL;
    }
    // As it currently stands the data behind m_startIndex is the previous data.
    // So we could seek backwards up to -m_startIndex.
    // We don't worry about this here, for simplicity sake.

    if (origin == SeekOrigin::End || origin == SeekOrigin::Start || offset < 0 ||
        offset >= Int64(getCount()))
    {
        // Empty the buffer
        _resetBuffer();
        // Seek on underlying stream
        return m_stream->seek(origin, offset);
    }

    // We can just seek on the buffered data
    consume(Index(offset));
    return SLANG_OK;
}

SlangResult BufferedReadStream::read(void* inBuffer, size_t length, size_t& outReadBytes)
{
    // If the buffer has no data and the read size is larger than the default read size - may as
    // well just read directly into the output buffer
    if (getCount() == 0 && length > m_defaultReadSize)
    {
        return m_stream->read(inBuffer, length, outReadBytes);
    }

    Byte* buffer = (Byte*)inBuffer;

    size_t totalReadBytes = 0;
    outReadBytes = 0;

    // Do a read to fill the buffer.
    SLANG_RETURN_ON_FAIL(update());

    while (length > 0)
    {
        const size_t bufferCount = size_t(getCount());

        if (bufferCount)
        {
            const size_t readCount = (bufferCount < length) ? bufferCount : length;

            ::memcpy(buffer, getBuffer(), readCount);

            consume(Index(readCount));
            buffer += readCount;
            length -= readCount;

            totalReadBytes += readCount;
        }
        else
        {
            if (m_stream == nullptr)
            {
                break;
            }

            // Read from underlying buffer
            size_t readBytes;
            SlangResult res = m_stream->read(buffer, length, readBytes);

            outReadBytes = totalReadBytes + readBytes;
            return res;
        }
    }

    outReadBytes = totalReadBytes;
    return SLANG_OK;
}

SlangResult BufferedReadStream::write(const void* buffer, size_t length)
{
    SLANG_UNUSED(buffer);
    SLANG_UNUSED(length);

    return SLANG_E_NOT_AVAILABLE;
}

bool BufferedReadStream::canRead()
{
    return getCount() > 0 || (m_stream && m_stream->canRead());
}

bool BufferedReadStream::canWrite()
{
    return false;
}

void BufferedReadStream::close()
{
    if (m_stream)
    {
        m_stream->close();
        m_stream.setNull();
    }
}

bool BufferedReadStream::isEnd()
{
    return getCount() == 0 && (m_stream == nullptr || m_stream->isEnd());
}

SlangResult BufferedReadStream::flush()
{
    return SLANG_E_NOT_AVAILABLE;
}

SlangResult BufferedReadStream::update()
{
    if (m_stream == nullptr)
    {
        // Should this return an error?
        return SLANG_OK;
    }

    // Repeat until we have enough space
    for (;;)
    {
        // How much buffer space do we have. We need at least m_defaultReadSize
        const size_t remainingCount = size_t(m_buffer.getCapacity() - m_buffer.getCount());

        if (remainingCount >= m_defaultReadSize)
        {
            break;
        }

        // If there is anything in the buffer shift it all down
        if (m_startIndex > 0)
        {
            Byte* buffer = m_buffer.getBuffer();
            const Index count = getCount();
            if (count > 0)
            {
                ::memmove(buffer, buffer + m_startIndex, count);
            }

            m_buffer.setCount(count);
            m_startIndex = 0;
        }
        else
        {
            // Make sure we have the space
            const Index prevCount = m_buffer.getCount();
            m_buffer.setCount(prevCount + m_defaultReadSize);
            m_buffer.setCount(prevCount);
        }
    }

    {
        const Index prevCount = m_buffer.getCount();
        m_buffer.setCount(prevCount + m_defaultReadSize);

        size_t readBytes = 0;

        const SlangResult res =
            m_stream->read(m_buffer.getBuffer() + prevCount, m_defaultReadSize, readBytes);

        m_buffer.setCount(prevCount + Index(readBytes));

        return res;
    }
}

SlangResult BufferedReadStream::readUntilContains(size_t size)
{
    while (true)
    {
        if (size_t(getCount()) >= size)
        {
            return SLANG_OK;
        }

        const size_t preCount = size_t(getCount());

        // Update buffer
        SLANG_RETURN_ON_FAIL(update());

        // If nothing was read yield
        if (preCount == getCount())
        {
            Process::sleepCurrentThread(0);
        }
    }
}


// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! StreamUtil !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult StreamUtil::readAndWrite(
    Stream* writeStream,
    ArrayView<Byte> bytesToWrite,
    Stream* readStream,
    List<Byte>& outReadBytes,
    Stream* errStream,
    List<Byte>& outErrBytes)
{
    std::thread writeThread(
        [&]()
        {
            writeStream->write(bytesToWrite.getBuffer(), (size_t)bytesToWrite.getCount());
            writeStream->close();
        });
    SlangResult readResult = SLANG_OK;
    std::thread readThread([&]() { readResult = readAll(readStream, 1024, outReadBytes); });
    std::thread readErrThread([&]() { readAll(errStream, 1024, outErrBytes); });
    writeThread.join();
    readThread.join();
    readErrThread.join();
    return readResult;
}

/* static */ SlangResult StreamUtil::readAll(Stream* stream, size_t readSize, List<Byte>& ioBytes)
{
    while (!stream->isEnd())
    {
        SLANG_RETURN_ON_FAIL(read(stream, readSize, ioBytes));
    }

    return SLANG_OK;
}

/* static */ SlangResult StreamUtil::read(Stream* stream, size_t readSize, List<Byte>& ioBytes)
{
    readSize = (readSize <= 0) ? 1024 : readSize;

    while (true)
    {
        const Index prevCount = ioBytes.getCount();
        ioBytes.setCount(prevCount + readSize);

        size_t readBytesCount;
        SLANG_RETURN_ON_FAIL(
            stream->read(ioBytes.getBuffer() + prevCount, readSize, readBytesCount));
        ioBytes.setCount(prevCount + Index(readBytesCount));

        if (readBytesCount == 0)
        {
            return SLANG_OK;
        }
    }
}

/* static */ SlangResult StreamUtil::discard(Stream* stream)
{
    Byte buf[1024];
    const Index bufSize = SLANG_COUNT_OF(buf);

    while (true)
    {
        size_t readBytesCount;
        SLANG_RETURN_ON_FAIL(stream->read(buf, bufSize, readBytesCount));

        if (readBytesCount == 0)
        {
            return SLANG_OK;
        }
    }
}

/* static */ SlangResult StreamUtil::discardAll(Stream* stream)
{
    while (!stream->isEnd())
    {
        SLANG_RETURN_ON_FAIL(discard(stream));
    }
    return SLANG_OK;
}


/* static */ SlangResult StreamUtil::readOrDiscard(
    Stream* stream,
    size_t readSize,
    List<Byte>* ioBytes)
{
    if (ioBytes)
    {
        return read(stream, readSize, *ioBytes);
    }
    else
    {
        return discard(stream);
    }
}

/* static */ SlangResult StreamUtil::readOrDiscardAll(
    Stream* stream,
    size_t readSize,
    List<Byte>* ioBytes)
{
    if (ioBytes)
    {
        return readAll(stream, readSize, *ioBytes);
    }
    else
    {
        return discardAll(stream);
    }
}

static FILE* _getFileFromStdStreamType(StdStreamType stdStream)
{
    switch (stdStream)
    {
    case StdStreamType::ErrorOut:
        return stderr;
    case StdStreamType::Out:
        return stdout;
    case StdStreamType::In:
        return stdin;
    default:
        return nullptr;
    }
}

static int _getBufferOptions(StreamBufferStyle style)
{
    switch (style)
    {
    case StreamBufferStyle::None:
        return _IONBF;
    case StreamBufferStyle::Line:
        return _IOLBF;
    default:
    case StreamBufferStyle::Full:
        return _IOFBF;
    }
}

/* static */ SlangResult StreamUtil::setStreamBufferStyle(
    StdStreamType stdStream,
    StreamBufferStyle style)
{
    FILE* file = _getFileFromStdStreamType(stdStream);

    if (file)
    {
        auto options = _getBufferOptions(style);

        // https://www.cplusplus.com/reference/cstdio/setvbuf/

        // NOTE! We don't set a buffer here (we pass in nullptr).
        // Passing nullptr is fine for 'no buffering' and sets a 'dynamic buffer' for others.
        // But it's not clear the behavior is around the buffer size. It seems the size is a
        // 'suggestion' so it will set the default but the documentation is unclear.
        if (setvbuf(file, nullptr, options, 0) == 0)
        {
            return SLANG_OK;
        }
        return SLANG_FAIL;
    }

    return SLANG_E_NOT_AVAILABLE;
}

} // namespace Slang
