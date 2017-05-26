#include "InitMethodFile.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <system_error>
#include <fstream>
#include <algorithm>

namespace {

void lockFile(int fd) {
  struct flock oflock;
  oflock.l_type = F_WRLCK; // write lock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // lock whole file

  // TODO: handle interrupts
  SYSCHECK(::fcntl(fd, F_SETLKW, &oflock));
}

void unlockFile(int fd) {
  struct flock oflock;
  oflock.l_type = F_UNLCK; // unlock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // unlock whole file

  SYSCHECK(::fcntl(fd, F_SETLKW, &oflock));
}

} // anonymous namespace


namespace thd {

InitMethodFile::InitMethodFile(std::string file_path, rank_type world_size)
 : _file_path(file_path)
 , _world_size(world_size)
{
  _file = ::open(_file_path.c_str(), O_RDWR | O_CREAT | O_EXCL, 0664);
  if (_file == -1 && errno == EEXIST) {
    _file = ::open(_file_path.c_str(), O_RDWR);
  }
  if (_file == -1) {
    throw std::system_error(_file, std::generic_category(), "cannot access '" + _file_path + "' file");
  }
}

InitMethodFile::~InitMethodFile() {
  ::close(_file);
}


InitMethod::Config InitMethodFile::getConfig() {
  InitMethod::Config config;
  lockFile(_file);

  std::fstream file(_file_path);
  std::string content{std::istreambuf_iterator<char>(file),
                      std::istreambuf_iterator<char>()};

  // rank is equal to number of lines inserted
  size_t rank = std::count(content.begin(), content.end(), '\n');
  config.rank = rank;
  if (config.rank == 0) {
    int listen_socket;
    std::string address;
    port_type port;
    std::tie(listen_socket, port) = listen();

    // pack message for other workers (we are first so we are master)
    file << port << '#';
    for (auto addr_str : getInterfaceAddresses()) {
        file << addr_str << ';';
    }
    file << std::endl;

    // TODO: connect here and recover your own address

    config.master = {
      .world_size = _world_size,
      .listen_socket = listen_socket,
      .listen_port = port,
    };
  } else {
    auto addr_end_pos = content.find('\n');
    if (addr_end_pos == std::string::npos)
      throw std::runtime_error("corrupted distributed init file");
    std::string master_info = content.substr(0, addr_end_pos);

    auto port_sep_pos = full_address.rfind('#');
    if (port_sep_pos == std::string::npos)
      throw std::runtime_error("corrupted distributed init file");

    std::string str_port = full_address.substr(0, port_sep_pos);
    auto port = convertToPort(std::stoul(str_port));

    std::vector<std::string> addresses;
    auto sep_pos = port_sep_pos;
    while (true) {
      auto next_sep_pos = full_address.find(';', sep_pos);
      if (next_sep_pos == std::string::npos) break;
      addresses.emplace_back(full_address.substr(sep_pos + 1, next_sep_pos);
      sep_pos = next_sep_pos;
    }

    // TODO: connect here and recover your own address

    config.worker = {
      .address = std::string(), // TODO
      .listen_port = port,
    };

    file << std::to_string(config.rank) << std::endl;
  }

  file.close();
  unlockFile(_file);

  if (config.rank == _world_size - 1) {
    ::remove(_file_path.c_str());
  }
  return config;
}

} // namespace thd
