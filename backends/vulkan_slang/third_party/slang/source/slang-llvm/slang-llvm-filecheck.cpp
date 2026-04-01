// This file contains a definition of LLVMFileCheck, an implementaion for
// IFileCheck.

#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang.h"

#include <core/slang-com-object.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/raw_ostream.h>
#include <slang-test/filecheck.h>

namespace slang_llvm
{

using namespace llvm;
using namespace Slang;

class LLVMFileCheck : IFileCheck, ComBaseObject
{
public:
    // ICastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) override;

    // IUnknown
    SLANG_COM_BASE_IUNKNOWN_ALL
    void* getInterface(const Guid& guid);
    void* getObject(const Guid& guid);

    // IFileCheck
    virtual TestResult SLANG_MCALL performTest(
        const char* programName,
        const char* rulesFilePath,
        const char* fileCheckPrefix,
        const char* stringToCheck,
        const char* stringToCheckName,
        ReportDiagnostic testReporter,
        void* reporterData,
        bool colorDiagnosticOutput) noexcept override;

private:
    // Everything we need to pass through LLVM back to our diagnostic handler
    struct ReporterData
    {
        ReportDiagnostic reportFun;
        // User data from the caller of performTest
        void* data;
        bool colorDiagnosticOutput;
        const char* programName;
        TestMessageType testMessageType;
    };

    static void fileCheckDiagHandler(const SMDiagnostic& diag, void* reporterData);
};

class DisplayedStringOStream : public raw_string_ostream
{
public:
    DisplayedStringOStream(std::string& s)
        : raw_string_ostream(s)
    {
    }
    virtual bool is_displayed() const override { return true; };
};

void LLVMFileCheck::fileCheckDiagHandler(const SMDiagnostic& diag, void* dataPtr)
{
    const ReporterData& reporterData = *reinterpret_cast<ReporterData*>(dataPtr);
    std::string s;
    DisplayedStringOStream o(s);
    o.enable_colors(reporterData.colorDiagnosticOutput);
    diag.print(reporterData.programName, o);
    reporterData.reportFun(reporterData.data, TestMessageType::TestFailure, s.c_str());
}

TestResult LLVMFileCheck::performTest(
    const char* const programName,
    const char* const rulesFilePath,
    const char* const fileCheckPrefix,
    const char* const stringToCheck,
    const char* const stringToCheckName,
    const ReportDiagnostic testReporter,
    void* const userReporterData,
    const bool colorDiagnosticOutput) noexcept
{
    //
    // Set up our FileCheck session
    //
    FileCheckRequest fcReq;
    fcReq.CheckPrefixes = {fileCheckPrefix};
    FileCheck fc(fcReq);

    //
    // Set up the LLVM source manager for diagnostic output from our input buffers
    //
    SourceMgr sourceManager;
    auto rulesTextOrError = MemoryBuffer::getFile(rulesFilePath, true);
    if (std::error_code err = rulesTextOrError.getError())
    {
        const std::string message = "Unable to load FileCheck rules file: " + err.message();
        testReporter(userReporterData, TestMessageType::RunError, message.c_str());
        return TestResult::Fail;
    }
    SmallString<4096> rulesBuffer;
    StringRef rulesStringRef = fc.CanonicalizeFile(*rulesTextOrError.get(), rulesBuffer);
    sourceManager.AddNewSourceBuffer(
        MemoryBuffer::getMemBuffer(rulesStringRef, rulesFilePath),
        SMLoc());

    SmallString<4096> inputBuffer;
    const auto inputStringMB =
        MemoryBuffer::getMemBuffer(StringRef(stringToCheck), stringToCheckName, false);
    const StringRef inputStringRef = fc.CanonicalizeFile(*inputStringMB.get(), inputBuffer);
    sourceManager.AddNewSourceBuffer(
        MemoryBuffer::getMemBuffer(inputStringRef, stringToCheckName),
        SMLoc());

    // Initialize this with a 'RunError' failure type. We'll "downgrade" this to
    // 'TestFailure' once we've done the FileCheck setup.
    ReporterData reporterData{
        testReporter,
        userReporterData,
        colorDiagnosticOutput,
        programName,
        TestMessageType::RunError};
    sourceManager.setDiagHandler(fileCheckDiagHandler, static_cast<void*>(&reporterData));

    auto checkPrefix = fc.buildCheckPrefixRegex();
    if (fc.readCheckFile(sourceManager, rulesStringRef, checkPrefix))
    {
        // FileCheck failed to find or understand any FileCheck rules in
        // the input file, automatic fail, and reported to the diag handler .
        return TestResult::Fail;
    }

    // We've done the FileCheck setup, so make sure that any diagnostics
    // reported on from here are just a regular test failure.
    reporterData.testMessageType = TestMessageType::TestFailure;
    if (!fc.checkInput(sourceManager, inputStringRef))
    {
        // An ordinary failure, the FileCheck rules didn't match
        return TestResult::Fail;
    }

    return TestResult::Pass;
}

void* LLVMFileCheck::castAs(const Guid& guid)
{
    if (auto ptr = getInterface(guid))
    {
        return ptr;
    }
    return getObject(guid);
}

void* LLVMFileCheck::getInterface(const Guid& guid)
{
    if (guid == ISlangUnknown::getTypeGuid() || guid == ICastable::getTypeGuid() ||
        guid == IFileCheck::getTypeGuid())
    {
        return static_cast<IFileCheck*>(this);
    }
    return nullptr;
}

void* LLVMFileCheck::getObject(const Guid& guid)
{
    SLANG_UNUSED(guid);
    return nullptr;
}

} // namespace slang_llvm

extern "C" SLANG_DLL_EXPORT SlangResult
createLLVMFileCheck_V1(const SlangUUID& intfGuid, void** out)
{
    Slang::ComPtr<slang_llvm::LLVMFileCheck> fileCheck(new slang_llvm::LLVMFileCheck);

    if (auto ptr = fileCheck->castAs(intfGuid))
    {
        fileCheck.detach();
        *out = ptr;
        return SLANG_OK;
    }

    return SLANG_E_NO_INTERFACE;
}
