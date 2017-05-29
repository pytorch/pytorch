#include "InitMethod.hpp"
#include "InitMethodUtils.hpp"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <system_error>
#include <thread>
#include <iterator>

namespace thd {
namespace init {

namespace {

void lockLoop(int fd, struct flock &oflock) {
  while (true) {
    int err = ::fcntl(fd, F_SETLKW, &oflock);
    if (err == 0) break;
    else if (err == EINTR) continue;
    else throw std::system_error(errno, std::system_category());
  }
}

void lockFile(int fd) {
  struct flock oflock;
  oflock.l_type = F_WRLCK; // write lock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // lock whole file
  lockLoop(fd, oflock);
}

void unlockFile(int fd) {
  struct flock oflock;
  oflock.l_type = F_UNLCK; // unlock
  oflock.l_whence = SEEK_SET;
  oflock.l_start = 0;
  oflock.l_len = 0; // unlock whole file
  lockLoop(fd, oflock);
}

} // anonymous namespace



InitMethod::Config initFile(std::string file_path, rank_type world_size, std::string group_name) {
  InitMethod::Config config;
  int fd;
  std::fstream file;
  std::string content;
  struct stat fd_stat, path_stat;

  // Loop until the file is either empty, or filled with ours group_name
  while (true) {
    // Loop until we have an open, locked and valid file
    while (true) {
      fd = ::open(file_path.c_str(), O_RDWR | O_CREAT, 0644);
      if (fd == -1) {
        throw std::system_error(fd, std::generic_category(),
                                "cannot access '" + file_path + "' file");
      }
      lockFile(fd);

      // This helps prevent a race when while we were waiting for the lock,
      // the file has been removed from the fs
      SYSCHECK(fstat(fd, &fd_stat));
      int err = stat(file_path.c_str(), &path_stat);
      if (err == 0 &&
          fd_stat.st_dev == path_stat.st_dev &&
          fd_stat.st_ino == path_stat.st_ino) {
        break;
      }
      ::close(fd);
    }

    file = std::fstream(file_path);
    content = {std::istreambuf_iterator<char>(file),
               std::istreambuf_iterator<char>()};

    if (content.length() == 0 || content.find(group_name) == 0) break;

    unlockFile(fd);
    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  // NOTE: the loop exits with a locked fd

  config.rank = std::count(content.begin(), content.end(), '\n');
  if (config.rank == 0) {
    int listen_socket;
    port_type port;
    std::tie(listen_socket, port) = listen();

    // pack message for other workers (we are first so we are master)
    file << group_name << ' ' << port << ' ';
    for (auto addr_str : getInterfaceAddresses()) {
        file << addr_str << ' ';
    }
    file << std::endl;

    file.close();
    unlockFile(fd);

    config.public_address = discoverWorkers(listen_socket, world_size);
    config.master = {
      .world_size = world_size,
      .listen_socket = listen_socket,
      .listen_port = port,
    };
  } else {
    file << std::endl; // reserve our rank
    file.seekp(0, std::ios_base::beg);

    std::string file_group_name;
    port_type port;
    file >> file_group_name >> port;
    std::vector<std::string> addresses =
        {std::istream_iterator<std::string>(file),
         std::istream_iterator<std::string>()};

    // Last member to join removes the file
    if (file_group_name != group_name) throw std::logic_error("file_group_name != group_name");
    if (config.rank == world_size - 1) {
      ::remove(file_path.c_str());
    }

    file.close();
    unlockFile(fd);

    std::string master_address;
    std::tie(master_address, config.public_address) = discoverMaster(addresses, port);
    config.worker = {
      .address = master_address,
      .port = port,
    };
  }

  return config;
}

} // namespace init
} // namespace thd
