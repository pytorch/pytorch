// test-process-main.cpp

#include "../../source/core/slang-http.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-string.h"
#include "../../source/core/slang-test-tool-util.h"
#include "slang-com-helper.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace TestProcess
{
using namespace Slang;

static void _makeStdStreamsUnbuffered()
{
    // Make these streams unbuffered
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
}

static SlangResult _outputCount(int argc, const char* const* argv)
{
    if (argc < 3)
    {
        return SLANG_FAIL;
    }
    // Get the count
    const Index count = stringToInt(argv[2]);

    // If we want to crash
    Index crashIndex = -1;
    if (argc > 3)
    {
        // When we crash, we want to make sure everything is flushed...
        _makeStdStreamsUnbuffered();
        crashIndex = stringToInt(argv[3]);
    }

    FILE* fileOut = stdout;

    StringBuilder buf;
    for (Index i = 0; i < count; ++i)
    {
        buf.clear();
        buf << i << "\n";

        fwrite(buf.getBuffer(), 1, buf.getLength(), fileOut);

        if (i == crashIndex)
        {
            // Use to simulate a crash.
            SLANG_BREAKPOINT(0);
            throw;
        }
    }

    // NOTE! There is no flush here, we want to see if everything works if we just stream out.
    return SLANG_OK;
}

static SlangResult _outputReflect()
{
    // Read lines from std input.
    // When hit line with just 'end', terminate

    // Get in as Stream

    RefPtr<Stream> stdinStream;
    Process::getStdStream(StdStreamType::In, stdinStream);

    FILE* fileOut = stdout;

    List<Byte> buffer;

    Index startIndex = 0;

    while (true)
    {
        SLANG_RETURN_ON_FAIL(StreamUtil::read(stdinStream, 0, buffer));

        while (true)
        {
            UnownedStringSlice slice(
                (const char*)buffer.begin() + startIndex,
                (const char*)buffer.end());
            UnownedStringSlice line;

            if (!StringUtil::extractLine(slice, line) || slice.begin() == nullptr)
            {
                break;
            }

            // Process the line
            if (line == UnownedStringSlice::fromLiteral("end"))
            {
                return SLANG_OK;
            }

            // Write the text to the output stream
            fwrite(line.begin(), 1, line.getLength(), fileOut);
            fputc('\n', fileOut);

            // Move the start index forward
            const Index newStartIndex = slice.begin()
                                            ? Index(slice.begin() - (const char*)buffer.getBuffer())
                                            : buffer.getCount();
            SLANG_ASSERT(newStartIndex > startIndex);
            startIndex = newStartIndex;
        }
    }
}

static SlangResult _httpReflect(int argc, const char* const* argv)
{
    SLANG_UNUSED(argc);
    SLANG_UNUSED(argv);

    RefPtr<Stream> stdinStream, stdoutStream;

    Process::getStdStream(StdStreamType::In, stdinStream);
    Process::getStdStream(StdStreamType::Out, stdoutStream);

    RefPtr<BufferedReadStream> readStream(new BufferedReadStream(stdinStream));

    RefPtr<HTTPPacketConnection> connection = new HTTPPacketConnection(readStream, stdoutStream);

    while (connection->isActive())
    {
        // Block waiting for content (or error/closed)
        SLANG_RETURN_ON_FAIL(connection->waitForResult());

        // If we have content do something with it
        if (connection->hasContent())
        {
            auto content = connection->getContent();

            // If it just holds 'end' then we are done
            const UnownedStringSlice slice((const char*)content.begin(), content.getCount());

            if (slice == UnownedStringSlice::fromLiteral("end"))
            {
                break;
            }

            // Else reflect it back
            SLANG_RETURN_ON_FAIL(connection->write(content.begin(), content.getCount()));

            // Consume that content/packet
            connection->consumeContent();
        }
    }

    return SLANG_OK;
}

static SlangResult execute(int argc, const char* const* argv)
{
    if (argc < 2)
    {
        return SLANG_FAIL;
    }

    // Get the tool name
    const String toolName = argv[1];
    if (toolName == "reflect")
    {
        return _outputReflect();
    }
    else if (toolName == "count")
    {
        return _outputCount(argc, argv);
    }
    else if (toolName == "http-reflect")
    {
        return _httpReflect(argc, argv);
    }
    return SLANG_E_NOT_AVAILABLE;
}

} // namespace TestProcess

int main(int argc, const char* const* argv)
{
    SlangResult res = TestProcess::execute(argc, argv);
    return (int)Slang::TestToolUtil::getReturnCode(res);
}
