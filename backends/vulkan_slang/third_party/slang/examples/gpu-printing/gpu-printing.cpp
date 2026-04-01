// gpu-printing.cpp
#include "gpu-printing.h"

#include <assert.h>
#include <string.h>

// This file implements the CPU side of a simple GPU printing
// library. The CPU code is responsible for scanning through
// buffers of "print commands" produced by GPU shaders, and
// executing those commands to print output.
//
// The opcodes for the printing commands are shared between
// CPU and GPU, and also between C++ and Slang, by putting
// their declarations in the `gpu-printing-ops.h` file
// and including them into both the host and device code
// to generate `enum` types.
//
enum class GPUPrintingOp : uint32_t
{
#define GPU_PRINTING_OP(NAME) NAME,
#include "gpu-printing-ops.h"
};

// One of the key ideas in this printing system is that strings
// are not encoded into the buffer of print commands directly,
// but are instead encoded using a hash of the string data.
//
// In order to map from a hash code back to the original string,
// the host side code for the printing system needs a way to
// pre-populate a lookup table with the strings that appear
// in a shader. The Slang reflection API provides a service to
// do exactly that.
//
void GPUPrinting::loadStrings(slang::ProgramLayout* slangReflection)
{
    // Given the Slang-generated reflection and layout information
    // for a program, we can query the number of string literals
    // that appear in the linked program.
    //
    SlangUInt hashedStringCount = slangReflection->getHashedStringCount();
    for (SlangUInt ii = 0; ii < hashedStringCount; ++ii)
    {
        // For each string we can fetch its bytes from the Slang
        // reflection data.
        //
        size_t stringSize = 0;
        char const* stringData = slangReflection->getHashedString(ii, &stringSize);

        // Then we can compute the hash code for that string using
        // another Slang API function.
        //
        // Note: the exact hashing algorithm that Slang uses for
        // string literals is not currently documented, and may
        // change in future releases of the compiler.
        //
        StringHash hash = spComputeStringHash(stringData, stringSize);

        // The `GPUPrinting` implementation will store the mapping
        // from hash codes back to strings in a simple STL `map`.
        //
        m_hashedStrings.insert(
            std::make_pair(hash, std::string(stringData, stringData + stringSize)));
    }
}

// The main service that the host code for the GPU printing library
// provides is a way to execute the printing commands that have been
// encoded to a buffer by shader code.
//
void GPUPrinting::processGPUPrintCommands(const void* data, size_t dataSize)
{
    // Everything that the GPU code writes to the buffer will be in
    // a granularity of 32-bits words, so we start by computing
    // how many words, total, will fit in the buffer.
    //
    uint32_t dataWordCount = uint32_t(dataSize / sizeof(uint32_t));
    //
    // If the buffer doesn't even have enough space for the leading counter,
    // then there is nothing to print.
    //
    if (dataWordCount < 1)
    {
        fprintf(stderr, "error: expected at least 4 bytes in GPU printing buffer\n");
        return;
    }
    //
    // Otherwise, we set ourselves up to start reading data from the buffer
    // at a granularity of 32-bit words.
    //
    const uint32_t* dataCursor = (const uint32_t*)data;

    // The first word of a printing buffer gives us the total number of
    // words that were appended by GPU printing operations.
    //
    uint32_t wordsAppended = *dataCursor++;
    //
    // Under normal operation, we will stop processing data from
    // the buffer after we have read everything the GPU wrote.
    //
    const uint32_t* dataEnd = dataCursor + wordsAppended;

    // If the number of bytes the GPU code tried to write (including
    // the counter stored in the first word of the buffer) exceeds what
    // the buffer could hold, then we will print a warning message,
    // indicating that the application might want to allocate a
    // larger buffer.
    //
    size_t totalBytesWritten = sizeof(uint32_t) * (wordsAppended + 1);
    if (totalBytesWritten > dataSize)
    {
        fprintf(
            stderr,
            "warning: GPU code attempted to write %llu bytes to the printing buffer, but only %llu "
            "bytes were available\n",
            (unsigned long long)totalBytesWritten,
            (unsigned long long)dataSize);

        // If the buffer is full, then we only want to read through
        // to the end of what is available.
        //
        dataEnd = ((const uint32_t*)data) + dataWordCount;
    }

    // We will now proceed to read off "commands" from the buffer,
    // and execute those commands to print things to `stdout`.
    //
    while (dataCursor < dataEnd)
    {
        // The first word of each command is encoded to hold both
        // an "opcode" for the command, and the number of "payload"
        // words that follow the header.
        //
        uint32_t cmdHeader = *dataCursor++;
        GPUPrintingOp op = GPUPrintingOp((cmdHeader >> 16) & 0xFFFF);
        uint32_t payloadWordCount = cmdHeader & 0xFFFF;

        // It is possible that we are at the end of the buffer,
        // and not all of the payload words could be written.
        // In such a case we will bail out of the printing loop to
        // avoid crashes from a command trying to fetch data past
        // the end of the buffer.
        //
        if (payloadWordCount > size_t(dataCursor - dataEnd))
        {
            break;
        }
        //
        // Otherwise, we can form a pointer to the payload words
        // for this command, and advance our cursor past the payload
        // to set up for reading the next command.
        //
        const uint32_t* payloadWords = dataCursor;
        const uint32_t* payloadWordsEnd = payloadWords + payloadWordCount;
        dataCursor += payloadWordCount;

        // What to do with a command depends a lot on which "op" was selected.
        switch (op)
        {
        default:
            // If we encounter an op that we don't understand, there is a change
            // that the buffer is corrupted or invalid, but we will try to
            // soldier on and process further commands.
            //
            fprintf(stderr, "error: unexpected GPU printing op %d\n", (int)op);
            break;

        case GPUPrintingOp::Nop:
            // The `Nop` case is a no-op, and allows GPU code to conservatively
            // allocate bytes in the printing buffer and then overwrite any
            // excess with zeros to trim their allocation.
            break;

        case GPUPrintingOp::NewLine:
            // The `NewLine` case prints a single '\n' and doesn't need any payload.
            putchar('\n');
            break;

            // Simple value printing cases can just load the bytes of
            // a value directly from the payload, and then print it.
            //
            // We will use a macro to avoid duplication the code shared
            // between these cases.
            //
#define CASE(OP, FORMAT, TYPE)                                              \
    case GPUPrintingOp::OP:                                                 \
        {                                                                   \
            TYPE value;                                                     \
            assert(payloadWordCount >= (sizeof(value) / sizeof(uint32_t))); \
            memcpy(&value, payloadWords, sizeof(value));                    \
            printf(FORMAT, value);                                          \
        }                                                                   \
        break

            CASE(Int32, "%d", int);
            CASE(UInt32, "%u", unsigned int);
            CASE(Float32, "%f", float);

#undef CASE

        case GPUPrintingOp::String:
            {
                // Strings are handled differently than other values because
                // most GPU graphics APIs do not natively support strings
                // in shader code.
                //
                // Instead, strings are handled by the printing logic in
                // terms of 32-bit hash codes. When printing a string,
                // the generated GPU code will write the hash value for
                // the string to the print buffer.
                //
                // On the CPU, we then read the hash code from the payload
                // for this command:
                //
                assert(payloadWordCount >= 1);
                StringHash hash = *payloadWords++;
                //
                // Next, we look up the hash value in a map from hash
                // codes to strings, that was seeded with strings known
                // to appear in the GPU code.
                //
                auto iter = m_hashedStrings.find(hash);
                if (iter == m_hashedStrings.end())
                {
                    // If we didn't have a string to match that hash code in
                    // our map, we can continue trying to print, but it is
                    // likely that the application code needs to be configured
                    // to pass in the right strings.
                    //
                    fprintf(stderr, "error: string with unknown hash 0x%x\n", hash);
                    continue;
                }

                // Once we've found a string that matches our hash
                // code, we can print it.
                //
                // TODO: This code isn't robust against strings with
                // embeded null bytes.
                // s
                printf("%s", iter->second.c_str());
            }
            break;

        case GPUPrintingOp::PrintF:
            {
                // Handling a general-purpose `printf` call requires looking
                // up the format string, and then processing further payload
                // words based on the format.
                //
                // Finding the format string follows logic similar to the
                // `GPUPrintingOp::String` case.
                //
                assert(payloadWords != payloadWordsEnd);
                StringHash formatHash = *payloadWords++;

                auto iter = m_hashedStrings.find(formatHash);
                if (iter == m_hashedStrings.end())
                {
                    // If we didn't have a string to match that hash code in
                    // our map, we can continue trying to print, but it is
                    // likely that the application code needs to be configured
                    // to pass in the right strings.
                    //
                    fprintf(stderr, "error: string with unknown hash 0x%x\n", formatHash);
                    continue;
                }
                std::string format = iter->second;

                // We can't just route things through to the `printf()` function
                // provided by standard library on the host CPU, because we don't
                // have a portable way to translate the payload data into
                // varargs that match the platform ABI.
                //
                // Instead, we have to scan through the string ourselves, and
                // implement a subset of the full `printf()`.
                //
                const char* cursor = format.c_str();
                const char* end = cursor + format.length();
                while (cursor != end)
                {
                    int c = *cursor++;

                    // If we see a byte other than `%`, then we can just
                    // output it directly and keep scanning the format string.
                    //
                    if (c != '%')
                    {
                        putchar(c);
                        continue;
                    }

                    // Otherwise, we have a `%` which is supposed to
                    // introduce a format specifier.
                    //
                    // If we are somehow at the end of the format
                    // string, then the format was bad.
                    //
                    if (cursor == end)
                    {
                        fprintf(stderr, "error: unexpected '%%' at and of format string\n");
                        break;
                    }

                    // If the next byte in the format string is
                    // the `%` character, then it is an escaped
                    // `%` so we should just emit it as-is and move along.
                    //
                    if (*cursor == '%')
                    {
                        putchar(*cursor++);
                        continue;
                    }

                    // TODO: For proper `printf()` support, we would need
                    // to read:
                    //
                    // * optional flags: `-+#0`
                    // * optional width specifier: a number or `*`
                    // * optional precision specifier: `.` and a number or `*`
                    // * optional length sub-specifiers: `h`, `l`, `ll`, etc.
                    //
                    // For now we ignore all those details and just
                    // read a single-byte specifier.
                    //
                    int specifier = *cursor++;
                    switch (specifier)
                    {
                    default:
                        fprintf(
                            stderr,
                            "error: unexpected format specifier '%c' (0x%X)\n",
                            specifier,
                            specifier);
                        break;


                        // When processing each format speecifier, we will
                        // read words from the payload, as necessary
                        // to yield a value of the expected type.
                        //
                        // To reduce the amount of boilerplate, we will
                        // use a macro to capture the shared code for
                        // common cases.
                        //
#define CASE(CHAR, FORMAT, TYPE)                              \
    case CHAR:                                                \
        {                                                     \
            assert(payloadWords != payloadWordsEnd);          \
            TYPE value;                                       \
            memcpy(&value, payloadWords, sizeof(value));      \
            payloadWords += sizeof(value) / sizeof(uint32_t); \
            printf(FORMAT, value);                            \
        }                                                     \
        break

                    case 'i': // `%i` is just an alias for `%d`
                        CASE('d', "%d", int);
                        CASE('u', "%u", unsigned int);
                        CASE('x', "%x", unsigned int);
                        CASE('X', "%X", unsigned int);

                        // Note: all of our printing support for floating-point
                        // values will use the `float` type instead of `double`.
                        // This isn't compatible with C rules, but makes more sense
                        // for GPU code.
                        //
                        CASE('f', "%f", float);
                        CASE('F', "%F", float);
                        CASE('e', "%e", float);
                        CASE('E', "%E", float);
                        CASE('g', "%g", float);
                        CASE('G', "%G", float);
                        CASE('c', "%c", int);

#undef CASE

                    case 's':
                        {
                            // The case for strings is more complicated
                            // just because it has to deal with our hashing
                            // scheme.
                            //
                            assert(payloadWords != payloadWordsEnd);
                            StringHash hash = *payloadWords++;
                            auto iter = m_hashedStrings.find(hash);
                            if (iter == m_hashedStrings.end())
                            {
                                fprintf(stderr, "error: string with unknown hash 0x%x\n", hash);
                                continue;
                            }
                            printf("%s", iter->second.c_str());
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
}
