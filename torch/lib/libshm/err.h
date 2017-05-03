#pragma once

#include <system_error>

#define SYSCHECK(call) { auto __ret = (call); if (__ret < 0) { throw std::system_error(errno, std::system_category()); } }
