#include "redis_store_handler.h"

#include <caffe2/core/logging.h>

#include <chrono>
#include <thread>
#include <vector>

namespace caffe2 {

RedisStoreHandler::RedisStoreHandler(
    std::string& host,
    int port,
    std::string& prefix)
    : host_(host), port_(port), prefix_(prefix) {
  struct timeval tv = {
      .tv_sec = 5,
      .tv_usec = 0,
  };

  redis_ = redisConnectWithTimeout(host.c_str(), port, tv);
  CAFFE_ENFORCE_NE(redis_, (redisContext*)nullptr);
  CAFFE_ENFORCE_EQ(redis_->err, 0, redis_->errstr);
}

RedisStoreHandler::~RedisStoreHandler() {
  redisFree(redis_);
}

std::string RedisStoreHandler::compoundKey(const std::string& name) {
  return prefix_ + name;
}

void RedisStoreHandler::set(const std::string& name, const std::string& data) {
  auto key = compoundKey(name);
  void* ptr = redisCommand(
      redis_,
      "SETNX %b %b",
      key.c_str(),
      (size_t)key.size(),
      data.c_str(),
      (size_t)data.size());
  CAFFE_ENFORCE_NE(ptr, (void*)nullptr, redis_->errstr);
  redisReply* reply = static_cast<redisReply*>(ptr);
  CAFFE_ENFORCE_EQ(reply->type, REDIS_REPLY_INTEGER);
  CAFFE_ENFORCE_EQ(
      reply->integer,
      1,
      "Value at ",
      name,
      " was already set",
      " (perhaps you reused a run ID you have used before?)");
}

std::string RedisStoreHandler::get(
    const std::string& name,
    const std::chrono::milliseconds& timeout) {
  // Block until key is set
  wait({name}, timeout);

  auto key = compoundKey(name);
  void* ptr = redisCommand(redis_, "GET %b", key.c_str(), (size_t)key.size());
  CAFFE_ENFORCE_NE(ptr, (void*)nullptr, redis_->errstr);
  redisReply* reply = static_cast<redisReply*>(ptr);
  CAFFE_ENFORCE_EQ(reply->type, REDIS_REPLY_STRING);
  return std::string(reply->str, reply->len);
}

int64_t RedisStoreHandler::add(const std::string& name, int64_t value) {
  auto key = compoundKey(name);
  void* ptr = redisCommand(
      redis_, "INCRBY %b %ld", key.c_str(), (size_t)key.size(), value);
  CAFFE_ENFORCE_NE(ptr, (void*)nullptr, redis_->errstr);
  redisReply* reply = static_cast<redisReply*>(ptr);
  CAFFE_ENFORCE_EQ(reply->type, REDIS_REPLY_INTEGER);
  return reply->integer;
}

bool RedisStoreHandler::check(const std::vector<std::string>& names) {
  std::vector<std::string> args;
  args.push_back("EXISTS");
  for (const auto& name : names) {
    args.push_back(compoundKey(name));
  }

  std::vector<const char*> argv;
  std::vector<size_t> argvlen;
  for (const auto& arg : args) {
    argv.push_back(arg.c_str());
    argvlen.push_back(arg.length());
  }

  auto argc = argv.size();
  void* ptr = redisCommandArgv(redis_, argc, argv.data(), argvlen.data());
  CAFFE_ENFORCE_NE(ptr, (void*)nullptr, redis_->errstr);
  redisReply* reply = static_cast<redisReply*>(ptr);
  CAFFE_ENFORCE_EQ(reply->type, REDIS_REPLY_INTEGER);
  return reply->integer == names.size();
}

void RedisStoreHandler::wait(
    const std::vector<std::string>& names,
    const std::chrono::milliseconds& timeout) {
  // Simple approach: poll...
  // Complex approach: use pub/sub.
  // Polling is fine for the typical rendezvous use case, as it is
  // only done at initialization time and  not at run time.
  const auto start = std::chrono::steady_clock::now();
  while (!check(names)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      STORE_HANDLER_TIMEOUT(
          "Wait timeout for name(s): ", c10::Join(" ", names));
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
} // namespace caffe2
