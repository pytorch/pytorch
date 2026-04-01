#include "output-stream.h"

#include "../util/record-utility.h"

namespace SlangRecord
{
FileOutputStream::FileOutputStream(const Slang::String& fileName, bool append)
{
    Slang::FileMode fileMode = append ? Slang::FileMode::Append : Slang::FileMode::Create;
    Slang::FileAccess fileAccess = Slang::FileAccess::Write;
    Slang::FileShare fileShare = Slang::FileShare::None;

    SlangResult res = m_fileStream.init(fileName, fileMode, fileAccess, fileShare);

    if (res != SLANG_OK)
    {
        SlangRecord::slangRecordLog(
            SlangRecord::LogLevel::Error,
            "Failed to open file %s\n",
            fileName.getBuffer());
        std::abort();
    }
}

FileOutputStream::~FileOutputStream()
{
    m_fileStream.close();
}

void FileOutputStream::write(const void* data, size_t len)
{
    SLANG_RECORD_CHECK(m_fileStream.write(data, len));
}

MemoryStream::MemoryStream()
    : m_memoryStream(Slang::FileAccess::Write)
{
}

void FileOutputStream::flush()
{
    SLANG_RECORD_CHECK(m_fileStream.flush());
}

void MemoryStream::write(const void* data, size_t len)
{
    SLANG_RECORD_CHECK(m_memoryStream.write(data, len));
}

void MemoryStream::flush()
{
    // This call will reset the underlying buffer to size 0,
    // and reset the write position to 0.
    m_memoryStream.setContent(nullptr, 0);
}
} // namespace SlangRecord
