#ifndef SLANG_CORE_WRITER_H
#define SLANG_CORE_WRITER_H

#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-list.h"
#include "slang-string.h"

#include <mutex>

namespace Slang
{

class WriterHelper
{
public:
    SLANG_ATTR_PRINTF(2, 3)
    SlangResult print(const char* format, ...);
    SlangResult put(const char* text);
    SlangResult put(const UnownedStringSlice& text);
    SLANG_FORCE_INLINE SlangResult write(const char* chars, size_t numChars)
    {
        return m_writer->write(chars, numChars);
    }
    SLANG_FORCE_INLINE void flush() { m_writer->flush(); }

    ISlangWriter* getWriter() const { return m_writer; }

    WriterHelper(ISlangWriter* writer)
        : m_writer(writer)
    {
    }

protected:
    ISlangWriter* m_writer;
};

struct WriterFlag
{
    enum Enum : uint32_t
    {
        IsStatic = 0x1,  ///< Means non ref counted
        IsConsole = 0x2, ///< True if console
        IsUnowned = 0x4, ///< True if doesn't own contained type
        AutoFlush = 0x8, ///< Automatically flushes after every call
    };

private:
    WriterFlag() = delete;
};
typedef uint32_t WriterFlags;

class BaseWriter : public ISlangWriter, public RefObject
{
public:
    // ISlangUnknown
    SLANG_REF_OBJECT_IUNKNOWN_QUERY_INTERFACE
    SLANG_REF_OBJECT_IUNKNOWN_ADD_REF
    SLANG_NO_THROW uint32_t SLANG_MCALL release() SLANG_OVERRIDE
    {
        return (m_flags & WriterFlag::IsStatic) ? (uint32_t)decreaseReference()
                                                : (uint32_t)releaseReference();
    }

    // ISlangWriter - default impl
    SLANG_NO_THROW virtual void SLANG_MCALL flush() SLANG_OVERRIDE {}
    SLANG_NO_THROW virtual bool SLANG_MCALL isConsole() SLANG_OVERRIDE
    {
        return (m_flags & WriterFlag::IsConsole) != 0;
    }
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL setMode(SlangWriterMode mode) SLANG_OVERRIDE
    {
        SLANG_UNUSED(mode);
        return SLANG_FAIL;
    }

    BaseWriter(WriterFlags flags)
        : m_flags(flags)
    {
    }

protected:
    ISlangUnknown* getInterface(const Guid& guid);
    WriterFlags m_flags;
};

/* Implemented the append buffer part of the writer, such that calls to begin/endAppendBuffer are
 * transformed into appropriate calls to write method */
class AppendBufferWriter : public BaseWriter
{
public:
    typedef BaseWriter Parent;

    // ISlangWriter default impl for appendBuffer
    SLANG_NO_THROW char* SLANG_MCALL beginAppendBuffer(size_t maxNumChars) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL endAppendBuffer(char* buffer, size_t numChars)
        SLANG_OVERRIDE;

    AppendBufferWriter(WriterFlags flags)
        : Parent(flags)
    {
    }

protected:
    List<char> m_appendBuffer;
    std::mutex mutex;
};

class CallbackWriter : public AppendBufferWriter
{
public:
    typedef AppendBufferWriter Parent;
    // ISlangWriter
    SLANG_NO_THROW char* SLANG_MCALL beginAppendBuffer(size_t maxNumChars) SLANG_OVERRIDE;
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL write(const char* chars, size_t numChars)
        SLANG_OVERRIDE;

    CallbackWriter(SlangDiagnosticCallback callback, const void* data, WriterFlags flags)
        : Parent(flags), m_callback(callback), m_data(data)
    {
    }

protected:
    SlangDiagnosticCallback m_callback;
    const void* m_data;
};

class FileWriter : public AppendBufferWriter
{
public:
    typedef AppendBufferWriter Parent;
    // ISlangWriter
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL write(const char* chars, size_t numChars)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual void SLANG_MCALL flush() SLANG_OVERRIDE;
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL setMode(SlangWriterMode mode) SLANG_OVERRIDE;

    static bool isFileConsole(FILE* file);
    static WriterFlags getDefaultFlags(FILE* file)
    {
        return isFileConsole(file) ? WriterFlags(WriterFlag::IsConsole) : 0;
    }

    /// Ctor
    FileWriter(FILE* file, WriterFlags flags)
        : Parent(flags | getDefaultFlags(file)), m_file(file)
    {
    }

    ///
    static SlangResult create(
        const char* filePath,
        const char* writeOptions,
        WriterFlags flags,
        ComPtr<ISlangWriter>& outWriter);

    static SlangResult createBinary(
        const char* filePath,
        WriterFlags flags,
        ComPtr<ISlangWriter>& outWriter)
    {
        return create(filePath, "wb", flags, outWriter);
    }
    static SlangResult createText(
        const char* filePath,
        WriterFlags flags,
        ComPtr<ISlangWriter>& outWriter)
    {
        return create(filePath, "w", flags, outWriter);
    }

    /// Dtor
    ~FileWriter();

protected:
    FILE* m_file;
};

class StringWriter : public BaseWriter
{
public:
    typedef BaseWriter Parent;
    // ISlangWriter
    SLANG_NO_THROW char* SLANG_MCALL beginAppendBuffer(size_t maxNumChars) SLANG_OVERRIDE;
    SLANG_NO_THROW SlangResult SLANG_MCALL endAppendBuffer(char* buffer, size_t numChars)
        SLANG_OVERRIDE;
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL write(const char* chars, size_t numChars)
        SLANG_OVERRIDE;

    /// Ctor
    StringWriter(StringBuilder* builder, WriterFlags flags)
        : Parent(flags), m_builder(builder)
    {
    }
    ~StringWriter() {}

protected:
    StringBuilder* m_builder;
};

class NullWriter : public AppendBufferWriter
{
public:
    typedef AppendBufferWriter Parent;
    // ISlangWriter
    SLANG_NO_THROW virtual SlangResult SLANG_MCALL write(const char* chars, size_t numChars)
        SLANG_OVERRIDE
    {
        SLANG_UNUSED(chars);
        SLANG_UNUSED(numChars);
        return SLANG_OK;
    }
    /// Ctor
    NullWriter(WriterFlags flags)
        : Parent(flags)
    {
    }
};

} // namespace Slang

#endif // SLANG_TEXT_WRITER_H
