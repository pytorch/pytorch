#define _CRT_SECURE_NO_WARNINGS

#include "slang-writer.h"

#include "slang-platform.h"
#include "slang-string-util.h"

// Includes to allow us to control console
// output when writing assembly dumps.
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <stdarg.h>

namespace Slang
{

/* !!!!!!!!!!!!!!!!!!!!!!!!! WriterHelper !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

SlangResult WriterHelper::print(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    SlangResult res = SLANG_OK;

    // numChars is the amount of characters needed *not* including terminating 0
    size_t numChars;
    {
        // Create a copy of args, as will be consumed by calcFormattedSize
        va_list argsCopy;
        va_copy(argsCopy, args);
        numChars = StringUtil::calcFormattedSize(format, argsCopy);
        va_end(argsCopy);
    }

    if (numChars > 0)
    {
        // We need to add 1 here, because calcFormatted, *requires* space for terminating 0
        char* appendBuffer = m_writer->beginAppendBuffer(numChars + 1);
        StringUtil::calcFormatted(format, args, numChars, appendBuffer);
        res = m_writer->endAppendBuffer(appendBuffer, numChars);
    }

    va_end(args);
    return res;
}

SlangResult WriterHelper::put(const char* text)
{
    return m_writer->write(text, ::strlen(text));
}

SlangResult WriterHelper::put(const UnownedStringSlice& text)
{
    return m_writer->write(text.begin(), text.getLength());
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! BaseWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

ISlangUnknown* BaseWriter::getInterface(const Guid& guid)
{
    return (guid == ISlangUnknown::getTypeGuid() || guid == ISlangWriter::getTypeGuid())
               ? static_cast<ISlangWriter*>(this)
               : nullptr;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! AppendBufferWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

SLANG_NO_THROW char* SLANG_MCALL AppendBufferWriter::beginAppendBuffer(size_t maxNumChars)
{
    mutex.lock();
    m_appendBuffer.setCount(maxNumChars);
    return m_appendBuffer.getBuffer();
}

SLANG_NO_THROW SlangResult SLANG_MCALL
AppendBufferWriter::endAppendBuffer(char* buffer, size_t numChars)
{
    SLANG_ASSERT(m_appendBuffer.getBuffer() == buffer && buffer + numChars <= m_appendBuffer.end());
    // Do the actual write
    SlangResult res = write(buffer, numChars);
    // Clear so that buffer can't be written from again without assert
    m_appendBuffer.clear();
    mutex.unlock();
    return res;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! CallbackWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

SLANG_NO_THROW char* SLANG_MCALL CallbackWriter::beginAppendBuffer(size_t maxNumChars)
{
    // Add one so there is always space for end termination, we need for the callback.
    m_appendBuffer.setCount(maxNumChars + 1);
    return m_appendBuffer.getBuffer();
}

SlangResult CallbackWriter::write(const char* chars, size_t numChars)
{
    if (numChars > 0)
    {
        char* appendBuffer = m_appendBuffer.getBuffer();
        // See if it's from an append buffer
        if (chars >= appendBuffer &&
            (chars + numChars) < (appendBuffer + m_appendBuffer.getCount()))
        {
            // Set terminating 0
            appendBuffer[(chars + numChars) - appendBuffer] = 0;

            m_callback(chars, (void*)m_data);
        }
        else
        {
            // Use the append buffer to add the terminating 0
            m_appendBuffer.setCount(numChars + 1);
            ::memcpy(m_appendBuffer.getBuffer(), chars, numChars);
            m_appendBuffer[numChars] = 0;

            m_callback(m_appendBuffer.getBuffer(), (void*)m_data);
        }
    }

    return SLANG_OK;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!! FileWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

FileWriter::~FileWriter()
{
    if (m_file)
    {
        ::fflush(m_file);

        if ((m_flags & WriterFlag::IsUnowned) == 0)
        {
            ::fclose(m_file);
        }
    }
}

SlangResult FileWriter::write(const char* text, size_t numChars)
{
    const size_t numWritten = ::fwrite(text, sizeof(char), numChars, m_file);
    if (m_flags & WriterFlag::AutoFlush)
    {
        ::fflush(m_file);
    }
    return numChars == numWritten ? SLANG_OK : SLANG_FAIL;
}

void FileWriter::flush()
{
    ::fflush(m_file);
}

/* static */ bool FileWriter::isFileConsole(FILE* file)
{
    const int stdoutFileDesc = _fileno(file);
    return _isatty(stdoutFileDesc) != 0;
}

SlangResult FileWriter::setMode(SlangWriterMode mode)
{
    switch (mode)
    {
    case SLANG_WRITER_MODE_BINARY:
        {
#ifdef _WIN32
            int stdoutFileDesc = _fileno(m_file);
            _setmode(stdoutFileDesc, _O_BINARY);
            return SLANG_OK;
#else
            break;
#endif
        }
    default:
        break;
    }
    return SLANG_FAIL;
}

/* static */ SlangResult FileWriter::create(
    const char* filePath,
    const char* writeOptions,
    WriterFlags flags,
    ComPtr<ISlangWriter>& outWriter)
{
    flags &= ~WriterFlag::IsUnowned;

    FILE* file = fopen(filePath, writeOptions);
    if (!file)
    {
        return SLANG_E_CANNOT_OPEN;
    }

    outWriter = new FileWriter(file, flags);
    return SLANG_OK;
}


/* !!!!!!!!!!!!!!!!!!!!!!!!! StringWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/

SLANG_NO_THROW char* SLANG_MCALL StringWriter::beginAppendBuffer(size_t maxNumChars)
{
    return m_builder->prepareForAppend(maxNumChars);
}

SLANG_NO_THROW SlangResult SLANG_MCALL StringWriter::endAppendBuffer(char* buffer, size_t numChars)
{
    m_builder->appendInPlace(buffer, numChars);
    return SLANG_OK;
}

SlangResult StringWriter::write(const char* chars, size_t numChars)
{
    if (numChars > 0)
    {
        m_builder->append(chars, numChars);
    }
    return SLANG_OK;
}

} // namespace Slang
