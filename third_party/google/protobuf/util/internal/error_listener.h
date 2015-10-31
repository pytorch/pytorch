// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_PROTOBUF_UTIL_CONVERTER_ERROR_LISTENER_H__
#define GOOGLE_PROTOBUF_UTIL_CONVERTER_ERROR_LISTENER_H__

#include <memory>
#ifndef _SHARED_PTR_H
#include <google/protobuf/stubs/shared_ptr.h>
#endif
#include <string>

#include <google/protobuf/stubs/callback.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/util/internal/location_tracker.h>
#include <google/protobuf/stubs/stringpiece.h>

namespace google {
namespace protobuf {
namespace util {
namespace converter {

// Interface for error listener.
class LIBPROTOBUF_EXPORT ErrorListener {
 public:
  virtual ~ErrorListener() {}

  // Reports an invalid name at the given location.
  virtual void InvalidName(const LocationTrackerInterface& loc,
                           StringPiece unknown_name, StringPiece message) = 0;

  // Reports an invalid value for a field.
  virtual void InvalidValue(const LocationTrackerInterface& loc,
                            StringPiece type_name, StringPiece value) = 0;

  // Reports a missing required field.
  virtual void MissingField(const LocationTrackerInterface& loc,
                            StringPiece missing_name) = 0;

 protected:
  ErrorListener() {}

 private:
  // Do not add any data members to this class.
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(ErrorListener);
};

// An error listener that ignores all errors.
class LIBPROTOBUF_EXPORT NoopErrorListener : public ErrorListener {
 public:
  NoopErrorListener() {}
  virtual ~NoopErrorListener() {}

  virtual void InvalidName(const LocationTrackerInterface& loc,
                           StringPiece unknown_name, StringPiece message) {}

  virtual void InvalidValue(const LocationTrackerInterface& loc,
                            StringPiece type_name, StringPiece value) {}

  virtual void MissingField(const LocationTrackerInterface& loc,
                            StringPiece missing_name) {}

 private:
  GOOGLE_DISALLOW_EVIL_CONSTRUCTORS(NoopErrorListener);
};


}  // namespace converter
}  // namespace util
}  // namespace protobuf

}  // namespace google
#endif  // GOOGLE_PROTOBUF_UTIL_CONVERTER_ERROR_LISTENER_H__
