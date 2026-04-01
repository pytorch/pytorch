#include "slang-text-io.h"

#include "slang-com-helper.h"

namespace Slang
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StreamWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SlangResult StreamWriter::init(const String& path, CharEncoding* encoding)
{
    RefPtr<FileStream> fileStream = new FileStream;
    SLANG_RETURN_ON_FAIL(fileStream->init(path, FileMode::Create));
    return init(fileStream, encoding);
}

SlangResult StreamWriter::init(RefPtr<Stream> stream, CharEncoding* encoding)
{
    m_stream = stream;
    m_encoding = encoding;
    if (encoding == CharEncoding::UTF16)
    {
        SLANG_RETURN_ON_FAIL(m_stream->write(&kUTF16Header, 2));
    }
    else if (encoding == CharEncoding::UTF16Reversed)
    {
        SLANG_RETURN_ON_FAIL(m_stream->write(&kUTF16ReversedHeader, 2));
    }

    return SLANG_OK;
}

SlangResult StreamWriter::writeSlice(const UnownedStringSlice& slice)
{
    // TODO(JS):
    // We can do better here. On Linux, this is a no-op and can just write directly (assuming slice
    // only contains \n)

    m_encodingBuffer.clear();

    StringBuilder sb;
#ifdef _WIN32
    const char newLine[] = "\r\n";
#else
    const char newLine[] = "\n";
#endif
    const Index length = slice.getLength();

    for (Index i = 0; i < length; i++)
    {
        if (slice[i] == '\r')
            sb << newLine;
        else if (slice[i] == '\n')
        {
            if (i > 0 && slice[i - 1] != '\r')
                sb << newLine;
        }
        else
            sb << slice[i];
    }

    // NOTE! This assumes that sb contains *complete* utf8 code points, which it might not, as
    // encoder is only able to handle complete code points.
    m_encodingBuffer.clear();
    m_encoding->encode(sb.getUnownedSlice(), m_encodingBuffer);
    return m_stream->write(m_encodingBuffer.getBuffer(), m_encodingBuffer.getCount());
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! StreamReader !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

StreamReader::StreamReader() {}

SlangResult StreamReader::init(const String& path)
{
    RefPtr<FileStream> fileStream = new FileStream;
    SLANG_RETURN_ON_FAIL(fileStream->init(path, FileMode::Open));
    return init(fileStream);
}

SlangResult StreamReader::init(RefPtr<Stream> stream, CharEncoding* encoding)
{
    m_stream = stream;
    m_encoding = encoding;
    SLANG_RETURN_ON_FAIL(readBuffer());

    if (encoding == nullptr)
    {
        size_t offset;
        m_encodingType = CharEncoding::determineEncoding(
            (const Byte*)m_buffer.getBuffer(),
            m_buffer.getCount(),
            offset);
        m_encoding = CharEncoding::getEncoding(m_encodingType);
        m_index = Index(offset);
    }
    else
    {
        m_encodingType = encoding->getEncodingType();
        m_encoding = encoding;
    }

    return SLANG_OK;
}

SlangResult StreamReader::readBuffer()
{
    m_buffer.setCount(0);
    m_index = 0;

    if (m_stream->isEnd())
    {
        return SLANG_OK;
    }

    m_buffer.setCount(4096);

    // TODO(JS): Not clear this is necessary
    memset(m_buffer.getBuffer(), 0, m_buffer.getCount() * sizeof(m_buffer[0]));

    size_t readBytes;
    SLANG_RETURN_ON_FAIL(m_stream->read(m_buffer.getBuffer(), m_buffer.getCount(), readBytes));

    m_buffer.setCount(Index(readBytes));
    m_index = 0;
    return SLANG_OK;
}

char StreamReader::readBufferChar()
{
    if (m_index < m_buffer.getCount())
    {
        return m_buffer[m_index++];
    }

    readBuffer();

    if (m_index < m_buffer.getCount())
    {
        return m_buffer[m_index++];
    }
    return 0;
}

SlangResult StreamReader::readToEnd(String& outString)
{
    StringBuilder sb(16384);
    while (!isEnd())
    {
        auto ch = read();
        if (isEnd())
            break;
        if (ch == '\r')
        {
            sb.append('\n');
            if (peek() == '\n')
                read();
        }
        else
            sb.append(ch);
    }

    outString = sb.produceString();
    return SLANG_OK;
}

} // namespace Slang
