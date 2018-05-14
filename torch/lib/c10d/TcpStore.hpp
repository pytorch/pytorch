#pragma once

#include "Store.hpp"
#include "Utils.hpp"

#include <memory>
#include <thread>
#include <unordered_map>


namespace c10d {

class TcpStoreDaemon {

 public:

  explicit TcpStoreDaemon(int storeListenSocket);
  ~TcpStoreDaemon();

  void join();

 protected:

  void run();
  void query(RankType rank);
  bool checkAndUpdate(std::vector<std::string>& keys) const;
  void wakeUpWaitingRanks(const std::string& key);

  std::thread daemonThread_;
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  std::unordered_map<std::string, std::vector<RankType>> waiting_;
  std::vector<size_t> keysAwaited_;
  std::vector<int> sockets_;

  int storeListenSocket_;
};

class TcpStore : public Store {

 public:

  explicit TcpStore(const std::string& masterAddr,
                    PortType masterPort,
                    bool isServer = false);

  virtual ~TcpStore();

  void set(
      const std::string& key,
      const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool check(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout = kDefaultTimeout) override;

 protected:

  bool isServer_;
  int storeSocket_ = -1;
  int masterListenSocket_ = -1;

  std::string tcpStoreAddr_;
  PortType tcpStorePort_;

  // Only needs to be launched on master rank
  std::unique_ptr<TcpStoreDaemon> tcpStoreDaemon_ = nullptr;
};

} // namespace c10d
