//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "Common.h"

#ifdef _WIN32

void ReadFile(std::vector<char>& out, const char* fileName)
{
    std::ifstream file(fileName, std::ios::ate | std::ios::binary);
    assert(file.is_open());
    size_t fileSize = (size_t)file.tellg();
    if(fileSize > 0)
    {
        out.resize(fileSize);
        file.seekg(0);
        file.read(out.data(), fileSize);
    }
    else
        out.clear();
}

void SetConsoleColor(CONSOLE_COLOR color)
{
    WORD attr = 0;
    switch(color)
    {
    case CONSOLE_COLOR::INFO:
        attr = FOREGROUND_INTENSITY;
        break;
    case CONSOLE_COLOR::NORMAL:
        attr = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE;
        break;
    case CONSOLE_COLOR::WARNING:
        attr = FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY;
        break;
    case CONSOLE_COLOR::ERROR_:
        attr = FOREGROUND_RED | FOREGROUND_INTENSITY;
        break;
    default:
        assert(0);
    }

    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(out, attr);
}

void PrintMessage(CONSOLE_COLOR color, const char* msg)
{
    if(color != CONSOLE_COLOR::NORMAL)
        SetConsoleColor(color);

    printf("%s\n", msg);

    if (color != CONSOLE_COLOR::NORMAL)
        SetConsoleColor(CONSOLE_COLOR::NORMAL);
}

void PrintMessage(CONSOLE_COLOR color, const wchar_t* msg)
{
    if(color != CONSOLE_COLOR::NORMAL)
        SetConsoleColor(color);

    wprintf(L"%s\n", msg);

    if (color != CONSOLE_COLOR::NORMAL)
        SetConsoleColor(CONSOLE_COLOR::NORMAL);
}

static const size_t CONSOLE_SMALL_BUF_SIZE = 256;

void PrintMessageV(CONSOLE_COLOR color, const char* format, va_list argList)
{
	size_t dstLen = (size_t)::_vscprintf(format, argList);
	if(dstLen)
	{
		bool useSmallBuf = dstLen < CONSOLE_SMALL_BUF_SIZE;
		char smallBuf[CONSOLE_SMALL_BUF_SIZE];
		std::vector<char> bigBuf(useSmallBuf ? 0 : dstLen + 1);
		char* bufPtr = useSmallBuf ? smallBuf : bigBuf.data();
		::vsprintf_s(bufPtr, dstLen + 1, format, argList);
		PrintMessage(color, bufPtr);
	}
}

void PrintMessageV(CONSOLE_COLOR color, const wchar_t* format, va_list argList)
{
	size_t dstLen = (size_t)::_vcwprintf(format, argList);
	if(dstLen)
	{
		bool useSmallBuf = dstLen < CONSOLE_SMALL_BUF_SIZE;
		wchar_t smallBuf[CONSOLE_SMALL_BUF_SIZE];
		std::vector<wchar_t> bigBuf(useSmallBuf ? 0 : dstLen + 1);
		wchar_t* bufPtr = useSmallBuf ? smallBuf : bigBuf.data();
		::vswprintf_s(bufPtr, dstLen + 1, format, argList);
		PrintMessage(color, bufPtr);
	}
}

void PrintMessageF(CONSOLE_COLOR color, const char* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(color, format, argList);
	va_end(argList);
}

void PrintMessageF(CONSOLE_COLOR color, const wchar_t* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(color, format, argList);
	va_end(argList);
}

void PrintWarningF(const char* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(CONSOLE_COLOR::WARNING, format, argList);
	va_end(argList);
}

void PrintWarningF(const wchar_t* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(CONSOLE_COLOR::WARNING, format, argList);
	va_end(argList);
}

void PrintErrorF(const char* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(CONSOLE_COLOR::WARNING, format, argList);
	va_end(argList);
}

void PrintErrorF(const wchar_t* format, ...)
{
	va_list argList;
	va_start(argList, format);
	PrintMessageV(CONSOLE_COLOR::WARNING, format, argList);
	va_end(argList);
}

void SaveFile(const wchar_t* filePath, const void* data, size_t dataSize)
{
    FILE* f = nullptr;
    _wfopen_s(&f, filePath, L"wb");
    if(f)
    {
        fwrite(data, 1, dataSize, f);
        fclose(f);
    }
    else
        assert(0);
}

std::wstring SizeToStr(size_t size)
{
    if(size == 0)
        return L"0";
    wchar_t result[32];
    double size2 = (double)size;
    if (size2 >= 1024.0*1024.0*1024.0*1024.0)
    {
        swprintf_s(result, L"%.2f TB", size2 / (1024.0*1024.0*1024.0*1024.0));
    }
    else if (size2 >= 1024.0*1024.0*1024.0)
    {
        swprintf_s(result, L"%.2f GB", size2 / (1024.0*1024.0*1024.0));
    }
    else if (size2 >= 1024.0*1024.0)
    {
        swprintf_s(result, L"%.2f MB", size2 / (1024.0*1024.0));
    }
    else if (size2 >= 1024.0)
    {
        swprintf_s(result, L"%.2f KB", size2 / 1024.0);
    }
    else
        swprintf_s(result, L"%llu B", size);
    return result;
}

bool ConvertCharsToUnicode(std::wstring *outStr, const std::string &s, unsigned codePage)
{
    if (s.empty())
    {
        outStr->clear();
        return true;
    }

    // Phase 1 - Get buffer size.
    const int size = MultiByteToWideChar(codePage, 0, s.data(), (int)s.length(), NULL, 0);
    if (size == 0)
    {
        outStr->clear();
        return false;
    }

    // Phase 2 - Do conversion.
    std::unique_ptr<wchar_t[]> buf(new wchar_t[(size_t)size]);
    int result = MultiByteToWideChar(codePage, 0, s.data(), (int)s.length(), buf.get(), size);
    if (result == 0)
    {
        outStr->clear();
        return false;
    }

    outStr->assign(buf.get(), (size_t)size);
    return true;
}

bool ConvertCharsToUnicode(std::wstring *outStr, const char *s, size_t sCharCount, unsigned codePage)
{
    if (sCharCount == 0)
    {
        outStr->clear();
        return true;
    }

    assert(sCharCount <= (size_t)INT_MAX);

    // Phase 1 - Get buffer size.
    int size = MultiByteToWideChar(codePage, 0, s, (int)sCharCount, NULL, 0);
    if (size == 0)
    {
        outStr->clear();
        return false;
    }

    // Phase 2 - Do conversion.
    std::unique_ptr<wchar_t[]> buf(new wchar_t[(size_t)size]);
    int result = MultiByteToWideChar(codePage, 0, s, (int)sCharCount, buf.get(), size);
    if (result == 0)
    {
        outStr->clear();
        return false;
    }

    outStr->assign(buf.get(), (size_t)size);
    return true;
}

const wchar_t* PhysicalDeviceTypeToStr(VkPhysicalDeviceType type)
{
    // Skipping common prefix VK_PHYSICAL_DEVICE_TYPE_
    static const wchar_t* const VALUES[] = {
        L"OTHER",
        L"INTEGRATED_GPU",
        L"DISCRETE_GPU",
        L"VIRTUAL_GPU",
        L"CPU",
    };
    return (uint32_t)type < _countof(VALUES) ? VALUES[(uint32_t)type] : L"";
}

const wchar_t* VendorIDToStr(uint32_t vendorID)
{
    switch(vendorID)
    {
    // Skipping common prefix VK_VENDOR_ID_ for these:
    case 0x10001: return L"VIV";
    case 0x10002: return L"VSI";
    case 0x10003: return L"KAZAN";
    case 0x10004: return L"CODEPLAY";
    case 0x10005: return L"MESA";
    case 0x10006: return L"POCL";
    // Others...
    case VENDOR_ID_AMD: return L"AMD";
    case VENDOR_ID_NVIDIA: return L"NVIDIA";
    case VENDOR_ID_INTEL: return L"Intel";
    case 0x1010: return L"ImgTec";
    case 0x13B5: return L"ARM";
    case 0x5143: return L"Qualcomm";
    }
    return L"";
}

#if VMA_VULKAN_VERSION >= 1002000
const wchar_t* DriverIDToStr(VkDriverId driverID)
{
    // Skipping common prefix VK_DRIVER_ID_
    static const wchar_t* const VALUES[] = {
        L"",
        L"AMD_PROPRIETARY",
        L"AMD_OPEN_SOURCE",
        L"MESA_RADV",
        L"NVIDIA_PROPRIETARY",
        L"INTEL_PROPRIETARY_WINDOWS",
        L"INTEL_OPEN_SOURCE_MESA",
        L"IMAGINATION_PROPRIETARY",
        L"QUALCOMM_PROPRIETARY",
        L"ARM_PROPRIETARY",
        L"GOOGLE_SWIFTSHADER",
        L"GGP_PROPRIETARY",
        L"BROADCOM_PROPRIETARY",
        L"MESA_LLVMPIPE",
        L"MOLTENVK",
    };
    return (uint32_t)driverID < _countof(VALUES) ? VALUES[(uint32_t)driverID] : L"";
}
#endif // #if VMA_VULKAN_VERSION >= 1002000


#endif // #ifdef _WIN32
