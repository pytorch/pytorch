#pragma once

#if defined(_WIN32)
#include <string>
#include <c10/util/win32-headers.h>
#include <c10/util/Exception.h>
#endif

namespace c10 {
#if defined(_WIN32)
inline std::wstring u8u16(const std::string& str) {
    if (str.empty()) {
        return std::wstring();
    }
    int size_needed = MultiByteToWideChar(
            CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), NULL, 0);
    TORCH_CHECK(size_needed > 0, "Error converting the content to Unicode");
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(
            CP_UTF8,
            0,
            str.c_str(),
            static_cast<int>(str.size()),
            &wstr[0],
            size_needed);
    return wstr;
}
#endif
}
