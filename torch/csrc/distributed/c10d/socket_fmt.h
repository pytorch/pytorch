// (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

/*
This file should not be included from other .h files and only used in cpp files
as it exposes the underlying platform specific socket headers.
*/

#include <string>

#ifdef _WIN32
#include <mutex>

#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netinet/in.h>
#endif

namespace c10d::detail {

// Returns a human-readable representation of the given socket address.
std::string formatSockAddr(const struct ::sockaddr* addr, socklen_t len);

} // namespace c10d::detail
