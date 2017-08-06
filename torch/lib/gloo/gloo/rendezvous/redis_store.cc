/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/rendezvous/redis_store.h"

#include <thread>

#include "gloo/common/error.h"
#include "gloo/common/logging.h"
#include "gloo/common/string.h"

namespace gloo {
namespace rendezvous {

static const std::chrono::seconds kWaitTimeout = std::chrono::seconds(60);

RedisStore::RedisStore(const std::string& host, int port) {
  struct timeval timeout = {.tv_sec = 2};
  redis_ = redisConnectWithTimeout(host.c_str(), port, timeout);
  GLOO_ENFORCE(redis_ != nullptr);
  if (redis_->err != 0) {
    GLOO_THROW_IO_EXCEPTION("Connecting to Redis: ", redis_->errstr);
  }
}

RedisStore::~RedisStore() {
  if (redis_ != nullptr) {
    redisFree(redis_);
  }
}

void RedisStore::set(const std::string& key, const std::vector<char>& data) {
  void* ptr = redisCommand(
      redis_,
      "SETNX %b %b",
      key.c_str(),
      (size_t)key.size(),
      data.data(),
      (size_t)data.size());
  if (ptr == nullptr) {
    GLOO_THROW_IO_EXCEPTION(redis_->errstr);
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (reply->type == REDIS_REPLY_ERROR) {
    GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
  }
  GLOO_ENFORCE_EQ(reply->type, REDIS_REPLY_INTEGER);
  GLOO_ENFORCE_EQ(reply->integer, 1, "Key '", key, "' already set");
  freeReplyObject(reply);
}

std::vector<char> RedisStore::get(const std::string& key) {
  // Block until key is set
  wait({key});

  // Get value
  void* ptr = redisCommand(redis_, "GET %b", key.c_str(), (size_t)key.size());
  if (ptr == nullptr) {
    GLOO_THROW_IO_EXCEPTION(redis_->errstr);
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (reply->type == REDIS_REPLY_ERROR) {
    GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
  }
  GLOO_ENFORCE_EQ(reply->type, REDIS_REPLY_STRING);
  std::vector<char> result(reply->str, reply->str + reply->len);
  freeReplyObject(reply);
  return result;
}

bool RedisStore::check(const std::vector<std::string>& keys) {
  std::vector<std::string> args;
  args.push_back("EXISTS");
  for (const auto& key : keys) {
    args.push_back(key);
  }

  std::vector<const char*> argv;
  std::vector<size_t> argvlen;
  for (const auto& arg : args) {
    argv.push_back(arg.c_str());
    argvlen.push_back(arg.length());
  }

  auto argc = argv.size();
  void* ptr = redisCommandArgv(redis_, argc, argv.data(), argvlen.data());
  if (ptr == nullptr) {
    GLOO_THROW_IO_EXCEPTION(redis_->errstr);
  }
  redisReply* reply = static_cast<redisReply*>(ptr);
  if (reply->type == REDIS_REPLY_ERROR) {
    GLOO_THROW_IO_EXCEPTION("Error: ", reply->str);
  }
  GLOO_ENFORCE_EQ(reply->type, REDIS_REPLY_INTEGER);
  auto result = (reply->integer == keys.size());
  freeReplyObject(reply);
  return result;
}

void RedisStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // Polling is fine for the typical rendezvous use case, as it is
  // only done at initialization time and  not at run time.
  const auto start = std::chrono::steady_clock::now();
  while (!check(keys)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != kNoTimeout && elapsed > timeout) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
          "Wait timeout for key(s): ", ::gloo::MakeString(keys)));
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // namespace rendezvous
} // namespace gloo
