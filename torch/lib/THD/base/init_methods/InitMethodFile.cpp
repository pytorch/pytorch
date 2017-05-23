#include "InitMethodFile.hpp"

#include <fcntl.h>

#include <cstdio>

namespace {

void appendString(FILE *file, std::string msg) {
  std::fseek(file, 0, SEEK_END);
  for (auto c : msg) {
    std::putc(c, file);
  }
}

void lockFile(FILE *file) {
  struct flock oflock;
  oflock.l_type = F_WRLCK; // write lock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // lock whole file

  int fd = ::fileno(file);
  SYSCHECK(::fcntl(fd, F_SETLKW, &oflock));
}

void unlockFile(FILE *file) {
  struct flock oflock;
  oflock.l_type = F_UNLCK; // unlock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // unlock whole file

  ::fflush(file);
  int fd = ::fileno(file);
  SYSCHECK(::fcntl(fd, F_SETLKW, &oflock));
}

} // anonymous namespace


namespace thd {

InitMethodFile::InitMethodFile(std::string file_path, rank_type world_size)
 : _file_path(file_path)
 , _world_size(world_size)
{
  _file = std::fopen(_file_path.c_str(), "rb+");
  if (!_file) {
    throw std::runtime_error("cannot access '" + _file_path + "' file");
  }
}

InitMethodFile::~InitMethodFile() {
  std::fclose(_file);
}


InitMethod::Config InitMethodFile::getConfig() {
  InitMethod::Config config;
  lockFile(_file);

  std::string content;
  int c; // NOTE: int, not char, required to handle EOF
  while ((c = std::fgetc(_file)) != EOF) {
    content += (char)c;
  }

  if (std::ferror(_file))
    throw std::runtime_error("unexpected error occured when reading '" + _file_path + "' file");

  size_t rank = std::count(content.begin(), content.end(), '\n'); // rank is equal to number of lines inserted
  config.rank = rank;
  if (config.rank == 0) {
    int listen_socket;
    std::string address;
    port_type port;
    std::tie(listen_socket, address, port) = listen();

    // pack message for other workers (we are first so we are master)
    std::string full_address = address + ";" + std::to_string(port) + "\n";
    appendString(_file, full_address);

    config.master = {
      .world_size = _world_size,
      .listen_socket = listen_socket,
      .listen_port = port,
    };
  } else {
    std::fseek(_file, 0, SEEK_SET);

    std::string full_address;
    while ((c = std::fgetc(_file)) != '\n') {
      full_address += c;
    }

    auto found_pos = full_address.rfind(";");
    if (found_pos == std::string::npos)
      throw std::runtime_error("something unexpected happened when reading a file, are you sure that the file was empty?");

    std::string str_port = full_address.substr(found_pos + 1);
    auto port = convertToPort(std::stoul(str_port));
    config.worker = {
      .address = full_address.substr(0, found_pos),
      .listen_port = port,
    };

    std::string rank_str = std::to_string(config.rank) + "\n";
    appendString(_file, rank_str);
  }

  unlockFile(_file);
  return config;
}

} // namespace thd
