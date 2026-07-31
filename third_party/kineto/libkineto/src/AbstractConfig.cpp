/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AbstractConfig.h"

#include <fmt/format.h>
#include <array>
#include <limits>
#include <sstream>

#include "Logger.h"

using namespace std::chrono;

using std::string;
using std::vector;

namespace KINETO_NAMESPACE {

constexpr char kWhitespace[] = "\t\n ";

static bool isWhitespace(string& s) {
  return s.find_first_not_of(kWhitespace) == string::npos;
}

// Remove whitespace from both end of string
static inline string trim(string& s) {
  if (s.empty()) {
    return s;
  } else if (isWhitespace(s)) {
    return "";
  }
  auto start = s.find_first_not_of(kWhitespace);
  auto end = s.find_last_not_of(kWhitespace);
  return s.substr(start, end - start + 1);
}

// Helper function for split.
// Return the index of char d in string s.
// If not found, returns the length of the string.
static int find(const char* s, char delim) {
  int i = 0;
  for (i = 0; s[i]; i++) {
    if (s[i] == delim) {
      break;
    }
  }
  return i;
}

// Split a string by delimiter
static vector<string> split(const string& s, char delim) {
  vector<string> res;
  const char* cs = s.c_str();
  for (int i = find(cs, delim); cs[i]; cs += i + 1, i = find(cs, delim)) {
    res.emplace_back(cs, i);
  }
  res.emplace_back(cs);
  return res;
}

// Remove a trailing comment.
static inline string stripComment(const string& s) {
  std::size_t pos = s.find('#');
  return s.substr(0, pos);
}

string AbstractConfig::toLower(string& s) const {
  string res = s;
  for (size_t i = 0; i < res.size(); i++) {
    if (res[i] >= 'A' && res[i] <= 'Z') {
      res[i] += ('a' - 'A');
    }
  }
  return res;
}

bool AbstractConfig::endsWith(const string& s, const string& suffix) const {
  return s.ends_with(suffix);
}

vector<string> AbstractConfig::splitAndTrim(const string& s, char delim) const {
  auto res = split(s, delim);
  for (string& x : res) {
    x = trim(x);
  }
  return res;
}

int64_t AbstractConfig::toIntRange(const string& val, int64_t min, int64_t max)
    const {
  char* invalid = nullptr;
  int64_t res = strtoll(val.c_str(), &invalid, 10);
  if (val.empty() || *invalid) {
    throw std::invalid_argument(fmt::format("Invalid integer: {}", val));
  } else if (res < min || res > max) {
    throw std::invalid_argument(fmt::format(
        "Invalid argument: {} - expected range [{}, {}]", res, min, max));
  }
  return res;
}

int32_t AbstractConfig::toInt32(const string& val) const {
  return static_cast<int32_t>(
      toIntRange(val, 0, std::numeric_limits<int32_t>::max()));
}

int64_t AbstractConfig::toInt64(const string& val) const {
  return toIntRange(val, 0, std::numeric_limits<int64_t>::max());
}

bool AbstractConfig::toBool(string& val) const {
  const std::array<string, 10> bool_vals{
      "n", "y", "no", "yes", "f", "t", "false", "true", "0", "1"};
  const string lower_val = toLower(val);
  for (size_t i = 0; i < bool_vals.size(); i++) {
    if (lower_val == bool_vals[i]) {
      return i % 2;
    }
  }
  throw std::invalid_argument(fmt::format("Invalid bool argument: {}", val));
}

bool AbstractConfig::parse(const string& conf) {
  std::istringstream iss(conf);
  string line;

  timestamp_ = system_clock::now();

  // Read the string stream 1 line at a time to parse.
  while (std::getline(iss, line)) {
    line = stripComment(line);
    if (isWhitespace(line)) {
      continue;
    }
    vector<string> key_val = splitAndTrim(line, '=');
    if (key_val.size() != 2) {
      LOG(ERROR) << "Invalid config line: " << line;
      return false;
    } else {
      bool handled = false;
      try {
        handled = handleOption(key_val[0], key_val[1]);
        if (!handled) {
          for (auto& feature_cfg : featureConfigs_) {
            if (feature_cfg.second->handleOption(key_val[0], key_val[1])) {
              handled = true;
              break;
            }
          }
        }
      } catch (const std::exception& e) {
        LOG(ERROR) << "Failed to parse config: " << e.what()
                   << " ; line: " << line;
        return false;
      }
      if (!handled) {
        // This might be due to using a newer config option on an
        // older binary where it is not supported. In this case,
        // print a warning message - but it is expected to work!
        LOG(WARNING) << "Unrecognized config line: " << line;
      }
    }
  }

  validate(timestamp_);

  // Store original text, used to detect updates
  source_ = conf;
  timestamp_ = system_clock::now();
  return true;
}

bool AbstractConfig::handleOption(
    const std::string& /* unused */,
    std::string& /* unused */) {
  // Unimplemented
  return false;
}

void AbstractConfig::printActivityProfilerConfig(std::ostream& s) const {
  for (const auto& feature_cfg : featureConfigs_) {
    feature_cfg.second->printActivityProfilerConfig(s);
  }
}

void AbstractConfig::setActivityDependentConfig() {
  for (const auto& feature_cfg : featureConfigs_) {
    feature_cfg.second->setActivityDependentConfig();
  }
}

} // namespace KINETO_NAMESPACE
