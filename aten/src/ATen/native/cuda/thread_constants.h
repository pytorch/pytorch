#pragma once
#include <c10/macros/Macros.h>

constexpr int num_threads() { return C10_WARP_SIZE * 4; }
constexpr int thread_work_size() { return 4; }
constexpr int block_work_size() { return thread_work_size() * num_threads(); }
