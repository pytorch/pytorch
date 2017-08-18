/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <set>
#include <string>
#include <vector>

namespace gloo {

const std::set<std::string>& kernelModules();

const int kPCIClass3D = 0x030200;
const int kPCIClassNetwork = 0x020000;

std::vector<std::string> pciDevices(int pciBusID);

int pciDistance(const std::string& a, const std::string& b);

const std::string& interfaceToBusID(const std::string& name);

int getInterfaceSpeedByName(const std::string& ifname);

const std::string& infinibandToBusID(const std::string& name);

} // namespace gloo
