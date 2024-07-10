#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <tuple>
#include <unordered_map>

#include <ATen/core/interned_strings.h>
#include <c10/util/thread_name.h>
#include <caffe2/utils/threadpool/WorkersPool.h>
#include <torch/csrc/distributed/c10d/control_plane/WorkerServer.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {
namespace control_plane {

namespace {
class RequestImpl : public Request {
 public:
  RequestImpl(const httplib::Request& req) : req_(req) {}

  const std::string& body() override {
    return req_.body;
  }

  const std::multimap<std::string, std::string>& params() const override {
    return req_.params;
  }

 private:
  const httplib::Request& req_;
};

class ResponseImpl : public Response {
 public:
  ResponseImpl(httplib::Response& res) : res_(res) {}

  void setStatus(int status) override {
    res_.status = status;
  }

  void setContent(std::string&& content, const std::string& content_type)
      override {
    res_.set_content(std::move(content), content_type);
  }

 private:
  httplib::Response& res_;
};

std::string jsonStrEscape(const std::string& str) {
  std::ostringstream ostream;
  for (char ch : str) {
    if (ch == '"') {
      ostream << "\\\"";
    } else if (ch == '\\') {
      ostream << "\\\\";
    } else if (ch == '\b') {
      ostream << "\\b";
    } else if (ch == '\f') {
      ostream << "\\f";
    } else if (ch == '\n') {
      ostream << "\\n";
    } else if (ch == '\r') {
      ostream << "\\r";
    } else if (ch == '\t') {
      ostream << "\\t";
    } else if ('\x00' <= ch && ch <= '\x1f') {
      ostream << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch);
    } else {
      ostream << ch;
    }
  }
  return ostream.str();
}
} // namespace

WorkerServer::WorkerServer(const std::string& hostOrFile, int port) {
  server_.Get("/", [](const httplib::Request& req, httplib::Response& res) {
    res.set_content(
        R"BODY(<h1>torch.distributed.WorkerServer</h1>
<a href="/handler/">Handler names</a>
)BODY",
        "text/html");
  });
  server_.Get(
      "/handler/", [](const httplib::Request& req, httplib::Response& res) {
        std::ostringstream body;
        body << "[";
        bool first = true;
        for (const auto& name : getHandlerNames()) {
          if (!first) {
            body << ",";
          }
          first = false;

          body << "\"" << jsonStrEscape(name) << "\"";
        }
        body << "]";

        res.set_content(body.str(), "application/json");
      });
  server_.Post(
      "/handler/:handler",
      [](const httplib::Request& req, httplib::Response& res) {
        auto handler_name = req.path_params.at("handler");
        HandlerFunc handler;
        try {
          handler = getHandler(handler_name);
        } catch (const std::exception& e) {
          res.status = 404;
          res.set_content(
              fmt::format("Handler {} not found: {}", handler_name, e.what()),
              "text/plain");
          return;
        }
        RequestImpl torchReq{req};
        ResponseImpl torchRes{res};

        try {
          handler(torchReq, torchRes);
        } catch (const std::exception& e) {
          res.status = 500;
          res.set_content(
              fmt::format("Handler {} failed: {}", handler_name, e.what()),
              "text/plain");
          return;
        } catch (...) {
          res.status = 500;
          res.set_content(
              fmt::format(
                  "Handler {} failed with unknown exception", handler_name),
              "text/plain");
          return;
        }
      });

  // adjust keep alives as it stops the server from shutting down quickly
  server_.set_keep_alive_timeout(1); // second, default is 5
  server_.set_keep_alive_max_count(
      30); // wait max 30 seconds before closing socket

  if (port == -1) {
    // using unix sockets
    server_.set_address_family(AF_UNIX);

    if (std::filesystem::exists(hostOrFile)) {
      throw std::runtime_error(fmt::format("{} already exists", hostOrFile));
    }

    C10D_WARNING("Server listening to UNIX {}", hostOrFile);
    if (!server_.bind_to_port(hostOrFile, 80)) {
      throw std::runtime_error(fmt::format("Error binding to {}", hostOrFile));
    }
  } else {
    C10D_WARNING("Server listening to TCP {}:{}", hostOrFile, port);
    if (!server_.bind_to_port(hostOrFile, port)) {
      throw std::runtime_error(
          fmt::format("Error binding to {}:{}", hostOrFile, port));
    }
  }

  serverThread_ = std::thread([this]() {
    c10::setThreadName("pt_workerserver");

    try {
      if (!server_.listen_after_bind()) {
        throw std::runtime_error("failed to listen");
      }
    } catch (std::exception& e) {
      C10D_ERROR("Error while running server: {}", e.what());
      throw;
    }
    C10D_WARNING("Server exited");
  });
}

void WorkerServer::shutdown() {
  C10D_WARNING("Server shutting down");
  server_.stop();
  serverThread_.join();
}

WorkerServer::~WorkerServer() {
  if (serverThread_.joinable()) {
    C10D_WARNING("WorkerServer destructor called without shutdown");
    shutdown();
  }
}

} // namespace control_plane
} // namespace c10d
