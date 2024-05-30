#pragma once

#include <functional>
#include <string>

#include <c10/macros/Export.h>

namespace c10d {
namespace control_plane {

// Request represents a request to the handler. This conceptually maps to an
// HTTP request but could be called via other transports.
class TORCH_API Request {
 public:
  virtual ~Request() = default;

  virtual const std::string& body() = 0;
};

// Response represents a response to the handler. This conceptually maps to an
// HTTP response but could be called via other transports.
class TORCH_API Response {
 public:
  virtual ~Response() = default;

  // Set the response body to the provided string.
  // TODO: add support for chunked responses
  virtual void setContent(
      std::string&& content,
      const std::string& content_type) = 0;

  // Set the response status code.
  // These should match standard HTTP status codes.
  virtual void setStatus(int status) = 0;
};

using HandlerFunc = std::function<void(const Request&, Response&)>;

// Registers a handler. The name needs to be unique and can be called by using
// getHandler directly or via WorkerServer for remote requests.
// These handlers are called from a background C++ thread concurrently with the
// main thread. These handlers need to be thread safe and not cause issues
// during Python training.
TORCH_API void registerHandler(const std::string& name, HandlerFunc f);

// Fetches a handler by name.
TORCH_API HandlerFunc getHandler(const std::string& name);

TORCH_API std::vector<std::string> getHandlerNames();

// Registers a handler statically.
// See registerHandler for more details.
class TORCH_API RegisterHandler {
 public:
  RegisterHandler(const std::string& name, HandlerFunc f) {
    registerHandler(name, f);
  }

  // disable move, copy
  RegisterHandler(const RegisterHandler&) = delete;
  RegisterHandler(RegisterHandler&&) = delete;
  RegisterHandler& operator=(const RegisterHandler&) = delete;
  RegisterHandler& operator=(RegisterHandler&&) = delete;
};

} // namespace control_plane
} // namespace c10d
