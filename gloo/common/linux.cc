/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/common/linux.h"

#include <dirent.h>
#include <errno.h>
#include <ifaddrs.h>
#include <linux/ethtool.h>
#include <linux/if.h>
#include <linux/sockios.h>
#include <linux/version.h>
#include <netdb.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <mutex>

#include "gloo/common/logging.h"

namespace gloo {

const std::set<std::string>& kernelModules() {
  static std::once_flag once;
  static std::set<std::string> modules;

  std::call_once(once, [](){
      std::ifstream ifs("/proc/modules");
      std::string line;
      while (std::getline(ifs, line)) {
        auto sep = line.find(' ');
        GLOO_ENFORCE_NE(sep, std::string::npos);
        modules.insert(line.substr(0, sep));
      }
    });

  return modules;
}

static const std::string kSysfsPath = "/sys/bus/pci/devices/";

static std::vector<std::string> listDir(const std::string& path) {
  DIR* dirp;
  struct dirent* dirent;
  std::vector<std::string> result;
  dirp = opendir(path.c_str());
  if (dirp == nullptr && errno == ENOENT) {
    // Ignore non-directories
    return result;
  }
  GLOO_ENFORCE(dirp != nullptr, strerror(errno));
  errno = 0;
  while ((dirent = readdir(dirp)) != nullptr) {
    if (dirent->d_name[0] == '.') {
      continue;
    }
    result.push_back(dirent->d_name);
  }
  GLOO_ENFORCE(errno == 0, strerror(errno));
  auto rv = closedir(dirp);
  GLOO_ENFORCE(rv == 0, strerror(errno));
  return result;
}

static unsigned int pciGetClass(const std::string& id) {
  auto path = kSysfsPath + id + "/class";
  std::ifstream ifs(path);
  GLOO_ENFORCE(ifs.good());
  unsigned int pciClass = 0;
  ifs.ignore(2);
  ifs >> std::hex >> pciClass;
  return pciClass;
}

std::vector<std::string> pciDevices(int pciClass) {
  std::vector<std::string> devices;
  for (const auto& device : listDir(kSysfsPath)) {
    if (pciClass != pciGetClass(device)) {
      continue;
    }

    devices.push_back(device);
  }
  return devices;
}

static std::string pciPath(const std::string& id) {
  auto path = kSysfsPath + id;
  std::transform(path.begin(), path.end(), path.begin(), ::tolower);
  std::array<char, 256> buf;
  auto rv = readlink(path.c_str(), buf.data(), buf.size());
  GLOO_ENFORCE_NE(rv, -1, strerror(errno));
  GLOO_ENFORCE_LT(rv, buf.size());
  return std::string(buf.data(), rv);
}

template<typename Out>
void split(const std::string& s, char delim, Out result) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}

int pciDistance(const std::string& a, const std::string& b) {
  std::vector<std::string> partsA;
  split(pciPath(a), '/', std::back_inserter(partsA));
  std::vector<std::string> partsB;
  split(pciPath(b), '/', std::back_inserter(partsB));

  // Count length of common prefix
  auto prefixLength = 0;
  for (;;) {
    if (prefixLength == partsA.size()) {
      break;
    }

    if (prefixLength == partsB.size()) {
      break;
    }

    if (partsA[prefixLength] != partsB[prefixLength]) {
      break;
    }

    prefixLength++;
  }

  return (partsA.size() - prefixLength) + (partsB.size() - prefixLength);
}

const std::string& interfaceToBusID(const std::string& name) {
  static std::once_flag once;
  static std::map<std::string, std::string> map;

  std::call_once(once, [](){
      for (const auto& device : pciDevices(kPCIClassNetwork)) {
        // Register interfaces for this devices
        const auto path = kSysfsPath + device + "/net";
        for (const auto& interface : listDir(path)) {
          map[interface] = device;
        }
      }
    });

  return map[name];
}

const std::string& infinibandToBusID(const std::string& name) {
  static std::once_flag once;
  static std::map<std::string, std::string> map;

  std::call_once(once, [](){
      for (const auto& device : pciDevices(kPCIClassNetwork)) {
        // Register interfaces for this devices
        const auto path = kSysfsPath + device + "/infiniband";
        for (const auto& interface : listDir(path)) {
          map[interface] = device;
        }
      }
    });

  return map[name];
}

static int getInterfaceSpeedGLinkSettings(int sock, struct ifreq* ifr) {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,6,0)
  constexpr auto link_mode_data_nwords = 3 * 127;
  struct {
    struct ethtool_link_settings req;
    __u32 link_mode_data[link_mode_data_nwords];
  } ecmd;
  int rv;

  ifr->ifr_data = &ecmd;
  memset(&ecmd, 0, sizeof(ecmd));
  ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;

  rv = ioctl(sock, SIOCETHTOOL, ifr);
  if (rv < 0 || ecmd.req.link_mode_masks_nwords >= 0) {
    return SPEED_UNKNOWN;
  }

  ecmd.req.cmd = ETHTOOL_GLINKSETTINGS;
  ecmd.req.link_mode_masks_nwords = -ecmd.req.link_mode_masks_nwords;
  rv = ioctl(sock, SIOCETHTOOL, ifr);
  if (rv < 0) {
    return SPEED_UNKNOWN;
  }

  return ecmd.req.speed;
#else
  (void)sock;
  (void)ifr;
  return SPEED_UNKNOWN;
#endif
}

static int getInterfaceSpeedGSet(int sock, struct ifreq* ifr) {
  struct ethtool_cmd edata;
  int rv;

  ifr->ifr_data = &edata;
  memset(&edata, 0, sizeof(edata));
  edata.cmd = ETHTOOL_GSET;

  rv = ioctl(sock, SIOCETHTOOL, ifr);
  if (rv < 0) {
    return SPEED_UNKNOWN;
  }

  return ethtool_cmd_speed(&edata);
}

int getInterfaceSpeedByName(const std::string& ifname) {
  int sock;
  struct ifreq ifr;
  int rv;
  size_t len;

  sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_IP);
  if (sock < 0) {
    return SPEED_UNKNOWN;
  }

  memset(&ifr, 0, sizeof(ifreq));
  len = ifname.length();
  len = std::min(len, sizeof(ifr.ifr_name) - 1);
  memcpy(ifr.ifr_name, ifname.c_str(), len);
  ifr.ifr_name[len] = '\0';

  rv = getInterfaceSpeedGLinkSettings(sock, &ifr);
  if (rv != SPEED_UNKNOWN) {
    close(sock);
    return rv;
  }
  rv = getInterfaceSpeedGSet(sock, &ifr);
  close(sock);

  return rv;
}

} // namespace gloo
