#include <c10/util/Unicode.h>

namespace c10 {
#if defined(_WIN32)
std::wstring u8u16(const std::string& str) {
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
std::string u16u8(const std::wstring& wstr) {
    if (wstr.empty()) {
        return std::string();
    }
    int size_needed = WideCharToMultiByte(
            CP_UTF8,
            0,
            wstr.c_str(),
            static_cast<int>(wstr.size()),
            NULL,
            0,
            NULL,
            NULL);
    TORCH_CHECK(size_needed > 0, "Error converting the content to UTF8");
    std::string str(size_needed, 0);
    WideCharToMultiByte(
            CP_UTF8,
            0,
            wstr.c_str(),
            static_cast<int>(wstr.size()),
            &str[0],
            size_needed,
            NULL,
            NULL);
    return str;
}
#endif
}  // namespace c10