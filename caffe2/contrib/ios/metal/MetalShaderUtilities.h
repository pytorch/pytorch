// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#import <string>
#import <sstream>
#import <functional>

std::string replace_first(std::string input, std::function<void(std::stringstream &fs)> fmt, int index);

#define REPLACE_CONSTANT(src, val, idx) { src = replace_first(src, [&](std::stringstream &fmt) { fmt << val; }, idx); }
