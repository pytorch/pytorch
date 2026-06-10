// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace torch::comms {

bool string_to_bool(std::string_view str);

// Convert environment variable to specified type, with default value if not set
template <typename T>
T env_to_value(std::string_view env_key, const T& default_value);

// Query rank and size based on TORCHCOMM_BOOTSTRAP_RANKSIZE_QUERY_METHOD
std::pair<int, int> query_ranksize();

} // namespace torch::comms
