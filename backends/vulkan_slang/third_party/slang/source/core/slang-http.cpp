#include "slang-http.h"

#include "slang-process.h"
#include "slang-string-util.h"

namespace Slang
{

static const UnownedStringSlice g_headerEnd = UnownedStringSlice::fromLiteral("\r\n\r\n");
static const UnownedStringSlice g_contentLength = UnownedStringSlice::fromLiteral("Content-Length");
static const UnownedStringSlice g_contentType = UnownedStringSlice::fromLiteral("Content-Type");

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! HTTPHeader !!!!!!!!!!!!!!!!!!!!!!! */

void HTTPHeader::reset()
{
    const UnownedStringSlice empty;

    m_contentLength = 0;
    m_mimeType = empty;
    m_encoding = empty;
    m_valuePairs.clear();
    m_header = empty;

    m_arena.deallocateAll();
}

/* static */ SlangResult HTTPHeader::readHeaderText(BufferedReadStream* stream, Index& outEndIndex)
{
    // https://microsoft.github.io/language-server-protocol/specifications/specification-current/

    while (true)
    {
        SLANG_RETURN_ON_FAIL(stream->update());

        const Index index = findHeaderEnd(stream);
        if (index >= 0)
        {
            outEndIndex = index;
            return SLANG_OK;
        }

        if (stream->isEnd())
        {
            return SLANG_FAIL;
        }

        Process::sleepCurrentThread(0);
    }
}

/* static */ Index HTTPHeader::findHeaderEnd(BufferedReadStream* stream)
{
    // This could be more efficient - it just searches until there are enough bytes to have
    // termination
    auto bytes = stream->getView();
    UnownedStringSlice input((const char*)bytes.begin(), (const char*)bytes.end());

    const Index index = input.indexOf(g_headerEnd);
    return (index >= 0) ? (index + g_headerEnd.getLength()) : index;
}

/* static */ SlangResult HTTPHeader::parse(const UnownedStringSlice& inSlice, HTTPHeader& out)
{
    out.reset();

    {
        auto slice = inSlice;
        // If has termination at end, remove so we don't have empty lines
        if (slice.endsWith(g_headerEnd))
        {
            slice = slice.head(slice.getLength() - g_headerEnd.getLength());
        }
        // Allocate on on the arena, so when we reference other slices, they are part of this
        // allocation.
        out.m_header = UnownedStringSlice(
            out.m_arena.allocateString(slice.begin(), slice.getLength()),
            slice.getLength());
    }

    // Okay, we need to split into lines, and then examine the contents
    for (auto line : LineParser(out.m_header))
    {
        // Examine the line for :
        Index index = line.indexOf(':');
        if (index < 0)
        {
            return SLANG_FAIL;
        }

        const UnownedStringSlice key = line.head(index).trim();
        const UnownedStringSlice value = line.tail(index + 1).trim();

        // Add the pair
        Pair pair{key, value};

        // We could check if key is already used. Some values can be repeated I believe.
        // So we just allow for now.

        out.m_valuePairs.add(pair);

        if (key == g_contentLength)
        {
            Index length;
            SLANG_RETURN_ON_FAIL(StringUtil::parseInt(value, length) || length < 0);

            out.m_contentLength = length;
        }
        else if (key == g_contentType)
        {
            List<UnownedStringSlice> slices;

            // text/html; charset=UTF-8
            StringUtil::split(value, ';', slices);

            if (slices.getCount() < 1)
            {
                return SLANG_FAIL;
            }
            // set the mime type
            out.m_mimeType = slices[0].trim();

            // Look for other parameters, in particular charset
            for (Index i = 1; i < slices.getCount(); ++i)
            {
                auto slice = slices[i];
                Index equalIndex = slice.indexOf('=');
                if (equalIndex >= 0)
                {
                    auto paramName = slice.head(equalIndex).trim();
                    auto paramValue = slice.tail(equalIndex + 1).trim();

                    if (paramName == UnownedStringSlice::fromLiteral("charset"))
                    {
                        out.m_encoding = paramValue;
                    }
                }
            }
        }
    }

    return SLANG_OK;
}

/* static */ SlangResult HTTPHeader::read(BufferedReadStream* stream, HTTPHeader& out)
{
    Index endIndex;
    SLANG_RETURN_ON_FAIL(readHeaderText(stream, endIndex));

    // Get header into a slice
    UnownedStringSlice headerText((const char*)stream->getBuffer(), endIndex);

    // Parse the slice into the out HttpHeader
    SLANG_RETURN_ON_FAIL(parse(headerText, out));

    // Can consume these bytes from the stream.
    stream->consume(endIndex);

    return SLANG_OK;
}

void HTTPHeader::append(StringBuilder& out) const
{
    // Output the content length
    out << g_contentLength << ": " << SlangSizeT(m_contentLength) << "\r\n";

    // If either is set construct a content type
    if (m_mimeType.getLength() || m_encoding.getLength())
    {
        out << g_contentType << ": ";

        // https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/MIME_types

        auto mimeType =
            m_mimeType.getLength() ? m_mimeType : UnownedStringSlice::fromLiteral("text/plain");
        auto encoding =
            m_encoding.getLength() ? m_encoding : UnownedStringSlice::fromLiteral("UTF-8");

        out << mimeType << "; ";
        out << "charset=" << encoding;

        out << "\r\n";
    }

    // Output any other data
    for (auto pair : m_valuePairs)
    {
        auto key = pair.key;
        // Ignore these types, as already output from data we already have
        if (key == g_contentType || key == g_contentLength)
        {
            continue;
        }

        out << key << ": " << pair.value << "\r\n";
    }

    // Add termination
    out << "\r\n";
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! HTTPPacketConnection !!!!!!!!!!!!!!!!!!!!!!! */

HTTPPacketConnection::HTTPPacketConnection(BufferedReadStream* readStream, Stream* writeStream)
    : m_readStream(readStream)
    , m_writeStream(writeStream)
    , m_readState(ReadState::Header)
    , m_readResult(SLANG_OK)
{
}

SlangResult HTTPPacketConnection::_handleHeader()
{
    SLANG_ASSERT(m_readState == ReadState::Header);

    const Index index = HTTPHeader::findHeaderEnd(m_readStream);
    if (index < 0)
    {
        // Don't have the full header yet
        return SLANG_OK;
    }

    // Okay we can parse the header
    UnownedStringSlice slice((const char*)m_readStream->getBuffer(), size_t(index));
    SLANG_RETURN_ON_FAIL(_updateReadResult(HTTPHeader::parse(slice, m_readHeader)));

    // Consume the header
    m_readStream->consume(index);

    // We are now consuming content
    m_readState = ReadState::Content;
    return SLANG_OK;
}

SlangResult HTTPPacketConnection::_handleContent()
{
    SLANG_ASSERT(m_readState == ReadState::Content);
    // Do we have enough content, mark as done
    if (m_readStream->getCount() >= m_readHeader.m_contentLength)
    {
        m_readState = ReadState::Done;
    }
    return SLANG_OK;
}

SlangResult HTTPPacketConnection::update()
{
    switch (m_readState)
    {
    case ReadState::Closed:
        return SLANG_OK;
    case ReadState::Error:
        return m_readResult;
    default:
        break;
    }

    SLANG_RETURN_ON_FAIL(_updateReadResult(m_readStream->update()));

    // Note will only indicate end if the buffer *and* backing stream are end/empty
    if (m_readStream->isEnd())
    {
        if (m_readState == ReadState::Header)
        {
            m_readState = ReadState::Closed;
        }
        else
        {
            // Closed without completing
            m_readState = ReadState::Error;
            m_readResult = SLANG_FAIL;
        }
        return SLANG_OK;
    }

    switch (m_readState)
    {
    case ReadState::Header:
        {
            SLANG_RETURN_ON_FAIL(_handleHeader());
            // We might be able to progress through content, if we have the header
            if (m_readState == ReadState::Content)
            {
                _handleContent();
            }
            break;
        }
    case ReadState::Content:
        {
            _handleContent();
            break;
        }
    default:
        break;
    }

    return m_readResult;
}


namespace
{ // anonymous

// Handles binary backoff like sleeping mechanism.
struct SleepState
{
    void sleep()
    {
        Process::sleepCurrentThread(m_intervalInMs);
        _update();
    }
    void reset()
    {
        m_intervalInMs = 0;
        m_count = 0;
    }
    void _update()
    {
        const Int maxIntervalInMs = 32;
        const Int initialCountThreshold = 4;

        ++m_count;

        const Int countThreshold = (m_intervalInMs == 0) ? initialCountThreshold : 1;

        // If we hit the count change the interval
        if (m_count >= countThreshold)
        {
            m_intervalInMs =
                (m_intervalInMs == 0) ? 1 : Math::Min(m_intervalInMs * 2, maxIntervalInMs);
            // Reset the count
            m_count = 0;
        }
    }

    Int m_intervalInMs = 0;
    Int m_count = 0;
};

} // namespace

SlangResult HTTPPacketConnection::waitForResult(Int timeOutInMs)
{
    m_readResult = SLANG_OK;

    int64_t startTick = 0;
    int64_t timeOutInTicks = -1;

    if (timeOutInMs >= 0)
    {
        timeOutInTicks = timeOutInMs * (Process::getClockFrequency() / 1000);
        startTick = Process::getClockTick();
    }

    SleepState sleepState;

    while (m_readState == ReadState::Header || m_readState == ReadState::Content)
    {
        const auto prevCount = m_readStream->getCount();

        SLANG_RETURN_ON_FAIL(update());

        if (m_readState == ReadState::Done)
        {
            break;
        }

        // We timed out
        if (timeOutInTicks >= 0 && int64_t(Process::getClockTick()) - startTick >= timeOutInTicks)
        {
            break;
        }

        if (prevCount == m_readStream->getCount())
        {
            sleepState.sleep();
        }
        else
        {
            sleepState.reset();
        }
    }

    return m_readResult;
}

void HTTPPacketConnection::consumeContent()
{
    SLANG_ASSERT(m_readState == ReadState::Done);
    if (m_readState == ReadState::Done)
    {
        // Consume the content
        m_readStream->consume(Index(m_readHeader.m_contentLength));
        // Back looking for the header again
        m_readState = ReadState::Header;
    }
}

SlangResult HTTPPacketConnection::write(const void* content, size_t sizeInBytes)
{
    // Write the header
    {
        HTTPHeader header;
        header.m_contentLength = sizeInBytes;

        StringBuilder buf;
        header.append(buf);

        SLANG_RETURN_ON_FAIL(m_writeStream->write(buf.getBuffer(), buf.getLength()));
    }

    // Write the content
    SLANG_RETURN_ON_FAIL(m_writeStream->write(content, sizeInBytes));

    return SLANG_OK;
}

} // namespace Slang
