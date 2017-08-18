#pragma once

#include <iostream>

#define HANDLE_EXCEPTIONS try {
#define END_HANDLE_EXCEPTIONS       \
} catch (std::exception &e) {       \
  THError(e.what());                \
}
