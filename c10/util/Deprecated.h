#pragma once

#if !defined(FBCODE_CAFFE2) && !defined(C10_NO_DEPRECATED)

#define C10_DEPRECATED [[deprecated]]
#define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;

#endif
