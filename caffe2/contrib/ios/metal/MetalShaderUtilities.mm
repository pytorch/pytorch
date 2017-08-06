// Copyright 2004-present Facebook. All Rights Reserved.

#import "MetalShaderUtilities.h"

std::string replace_first(std::string input, std::function<void(std::stringstream &fs)> fmt, int index) {
  std::stringstream value_format_stream;
  value_format_stream << "= ";
  fmt(value_format_stream);

  std::stringstream token_format_stream;
  token_format_stream << "[[ function_constant(" << index << ") ]]";

  size_t position = input.find(token_format_stream.str());
  if (position != std::string::npos) {
    size_t length = token_format_stream.str().length();
    return input.replace(position, length, value_format_stream.str());
  } else {
    return input;
  }
}
