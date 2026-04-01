// unit-test-process.cpp

#include "../../source/core/slang-http.h"
#include "../../source/core/slang-io.h"
#include "../../source/core/slang-process-util.h"
#include "../../source/core/slang-random-generator.h"
#include "../../source/core/slang-string-util.h"
#include "unit-test/slang-unit-test.h"

using namespace Slang;

static SlangResult _createProcess(
    UnitTestContext* context,
    const char* toolName,
    const List<String>* optArgs,
    RefPtr<Process>& outProcess)
{
    CommandLine cmdLine;
    cmdLine.setExecutableLocation(ExecutableLocation(context->executableDirectory, "test-process"));
    cmdLine.addArg(toolName);
    if (optArgs)
    {
        cmdLine.m_args.addRange(optArgs->getBuffer(), optArgs->getCount());
    }

    SLANG_RETURN_ON_FAIL(Process::create(cmdLine, Process::Flag::AttachDebugger, outProcess));

    return SLANG_OK;
}

static SlangResult _httpReflectTest(UnitTestContext* context)
{
    SlangResult finalRes = SLANG_OK;

    RefPtr<Process> process;
    SLANG_RETURN_ON_FAIL(_createProcess(context, "http-reflect", nullptr, process));

    Stream* writeStream = process->getStream(StdStreamType::In);
    RefPtr<BufferedReadStream> readStream(
        new BufferedReadStream(process->getStream(StdStreamType::Out)));
    RefPtr<HTTPPacketConnection> connection = new HTTPPacketConnection(readStream, writeStream);

    RefPtr<RandomGenerator> rand = RandomGenerator::create(10000);

    for (Index i = 0; i < 100; i++)
    {
        if (process->isTerminated())
        {
            return SLANG_FAIL;
        }

        const Index size = Index(rand->nextInt32UpTo(8192));

        List<Byte> buf;
        buf.setCount(size);
        // Build up a buffer
        rand->nextData(buf.getBuffer(), size_t(size));

        // Write the data
        SLANG_RETURN_ON_FAIL(connection->write(buf.getBuffer(), size_t(size)));

        // Wait for the response
        SLANG_RETURN_ON_FAIL(connection->waitForResult());

        // If we don't have content then something has gone wrong
        if (!connection->hasContent())
        {
            finalRes = SLANG_FAIL;
            break;
        }

        // Check the content is the same
        auto readContent = connection->getContent();
        if (readContent != buf.getArrayView())
        {
            finalRes = SLANG_FAIL;
            break;
        }

        // Consume that packet
        connection->consumeContent();
    }

    // Send the end
    {
        const char end[] = "end";
        SLANG_RETURN_ON_FAIL(connection->write(end, SLANG_COUNT_OF(end) - 1));
    }

    process->waitForTermination();
    return finalRes;
}

static SlangResult _countTest(UnitTestContext* context, Index size, Index crashIndex = -1)
{
    /* Here we are trying to test what happens if the server produces a large amount of data, and
    we just wait for termination. Do we receive all of the data irrespective of how much there is?
  */

    List<String> args;
    {
        StringBuilder buf;
        buf << size;

        args.add(buf);

        if (crashIndex >= 0)
        {
            buf.clear();
            buf << crashIndex;
            args.add(buf);
        }
    }

    RefPtr<Process> process;
    SLANG_RETURN_ON_FAIL(_createProcess(context, "count", &args, process));

    ExecuteResult exeRes;

#if 0
    /* It does block on ~4k of data which matches up with the buffer size, using this mechanism only works up to 4k on windows
    which matches the default pipe buffer size */
    process->waitForTermination();
#endif

    SLANG_RETURN_ON_FAIL(ProcessUtil::readUntilTermination(process, exeRes));

    Index v = 0;
    for (auto line : LineParser(exeRes.standardOutput.getUnownedSlice()))
    {
        if (line.getLength() == 0)
        {
            continue;
        }

        Index value;
        StringUtil::parseInt(line, value);

        if (value != v)
        {
            return SLANG_FAIL;
        }

        v++;
    }

    const Index endIndex = (crashIndex >= 0) ? (crashIndex + 1) : size;

    return v == endIndex ? SLANG_OK : SLANG_FAIL;
}

static SlangResult _countTests(UnitTestContext* context)
{
    const Index sizes[] = {1, 10, 1000, 1000, 10000, 100000};
    for (auto size : sizes)
    {
        SLANG_RETURN_ON_FAIL(_countTest(context, size));
        SLANG_RETURN_ON_FAIL(_countTest(context, size, size / 2));
    }

    return SLANG_OK;
}

static SlangResult _reflectTest(UnitTestContext* context)
{
    RefPtr<Process> process;
    SLANG_RETURN_ON_FAIL(_createProcess(context, "reflect", nullptr, process));

    // Write a bunch of stuff to the stream
    Stream* readStream = process->getStream(StdStreamType::Out);
    Stream* writeStream = process->getStream(StdStreamType::In);

    List<Byte> readBuffer;

    for (Index i = 0; i < 10000; i++)
    {
        SLANG_RETURN_ON_FAIL(StreamUtil::read(readStream, 0, readBuffer));

        StringBuilder buf;

        buf << i << " Hello " << i << "\n";
        SLANG_RETURN_ON_FAIL(writeStream->write(buf.getBuffer(), buf.getLength()));
    }

    const char end[] = "end\n";
    SLANG_RETURN_ON_FAIL(writeStream->write(end, SLANG_COUNT_OF(end) - 1));
    writeStream->flush();

    SLANG_RETURN_ON_FAIL(StreamUtil::readAll(readStream, 0, readBuffer));

    return SLANG_OK;
}

SLANG_UNIT_TEST(CommandLineProcess)
{
    SLANG_CHECK(SLANG_SUCCEEDED(_countTests(unitTestContext)));
    SLANG_CHECK(SLANG_SUCCEEDED(_reflectTest(unitTestContext)));
    SLANG_CHECK(SLANG_SUCCEEDED(_httpReflectTest(unitTestContext)));
}
