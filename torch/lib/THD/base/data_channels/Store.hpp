#pragma once

#include "../ChannelUtils.hpp"
#include "gloo/rendezvous/store.h"

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace thd {

class Store : public ::gloo::rendezvous::Store {
  public:
    Store();
    ~Store();

    void set(const std::string& key, const std::vector<char>& data) override;
    std::vector<char> get(const std::string& key) override; 
    void wait(const std::vector<std::string>& keys) override;

  private:
    rank_type _rank;
    int _socket;
    std::string _store_addr;
    port_type _store_port;
    std::thread _store_thread; // it is initialised only in a selected process
};

} // namespace thd
