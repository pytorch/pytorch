#pragma once

#include "../ChannelUtils.hpp"
#include "gloo/rendezvous/store.h"

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace thd {

struct Store : public ::gloo::rendezvous::Store {
private:
  struct StoreDeamon {
    StoreDeamon() = delete;
    StoreDeamon(int listen_socket, rank_type world_size);
    ~StoreDeamon();

    void join();

  private:
    using store_type = std::unordered_map<std::string, std::vector<char>>;

    void deamon();
    void query(rank_type rank);
    bool checkAndUpdate(std::vector<std::string>& keys) const;

    int _listen_socket;

    std::thread _deamon;
    store_type _store;
    std::unordered_map<std::string, std::vector<rank_type>> _waiting;
    std::vector<std::size_t> _keys_awaited;
    std::vector<int> _sockets;
  };

public:
  Store(rank_type rank, int listen_socket, const std::string& addr,
        port_type port, rank_type world_size);
  ~Store();

  void set(const std::string& key, const std::vector<char>& data) override;
  std::vector<char> get(const std::string& key) override;
  void wait(const std::vector<std::string>& keys) override;

private:
  rank_type _rank;
  int _listen_socket;
  int _socket;
  std::string _store_addr;
  port_type _store_port;
  std::unique_ptr<StoreDeamon> _store_thread; // it is initialised only in a selected process
};

} // namespace thd
