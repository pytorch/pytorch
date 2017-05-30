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



InitMethod::Config initFile(std::string file_path, rank_type world_size,
                            std::string group_name, int assigned_rank) {
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
      SYSCHECK(::fstat(fd, &fd_stat));
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

  // Remember our order number.
  std::size_t order = std::count(content.begin(), content.end(), '\n');

  int listen_socket;
  port_type port;
  std::tie(listen_socket, port) = listen();

  auto ifa_addresses = getInterfaceAddresses();
  file << group_name << ' ' << assigned_rank << ' ' << port
       << ' ' << ifa_addresses.size();
  for (auto addr_str : ifa_addresses) {
    file << ' ' << addr_str;
  }
  file << std::endl;

  std::size_t lines = 0; // we have just added new line
  while (lines < world_size) { // wait until all processes will write their info
    unlockFile(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    lockFile(fd);

    file.sync();
    file.seekp(0, std::ios_base::beg);
    content = {std::istreambuf_iterator<char>(file),
               std::istreambuf_iterator<char>()};
    lines = std::count(content.begin(), content.end(), '\n');
    file.seekp(0, std::ios_base::beg);
  }
  // NOTE: the loop exits with a locked fd

  port_type master_port;
  std::vector<std::string> master_addresses;
  std::vector<int> ranks(world_size);
  for (std::size_t i = 0; i < world_size; ++i) {
    std::string file_group_name, tmp_address;
    std::size_t addresses_count;
    int rank;
    port_type tmp_port;

    file >> file_group_name >> rank >> tmp_port >> addresses_count;
    if (file_group_name != group_name) {
      throw std::logic_error("file_group_name != group_name");
    }

    std::vector<std::string> tmp_addresses(addresses_count);
    for (std::size_t j = 0; j < addresses_count; ++j) {
      file >> tmp_address;
      tmp_addresses.emplace(tmp_addresses.begin() + j, tmp_address);
    }

    ranks[i] = rank;
    // Whether there is already assigned rank 0, or we have to get addresses and port
    // from first process which has unassigned rank (it will be rank 0).
    if (rank == 0 || (rank < 0 && master_addresses.size() == 0)) {
      master_port = tmp_port;
      master_addresses = tmp_addresses;
    }
  }

  if (assigned_rank >= 0) {
    config.rank = assigned_rank;
  } else {
    // Calculate how many unassigned there was before us (including us).
    std::size_t unassigned = 1 + std::count_if(ranks.begin(), ranks.begin() + order,
                                               [](int rank) { return rank < 0; });

    // Calculate actual rank by finding `unassigned` number of empty ranks.
    for (std::size_t rank = 0; rank < world_size && unassigned > 0; ++rank) {
      if (std::find(ranks.begin(), ranks.end(), rank) == ranks.end()) {
        unassigned--;
      }

      if (unassigned == 0) config.rank = rank;
    }
  }

  file << std::endl; // reserve our rank

  // Member which is last to use the file has to remove it.
  if (lines == 2 * world_size - 1) {
    ::remove(file_path.c_str());
  }

  file.close();
  unlockFile(fd);

  if (config.rank == 0) {
    config.public_address = discoverWorkers(listen_socket, world_size);
    config.master = {
      .world_size = world_size,
      .listen_socket = listen_socket,
      .listen_port = master_port,
    };
  } else {
    ::close(listen_socket);

    std::string master_address;
    std::tie(master_address, config.public_address) = discoverMaster(master_addresses, master_port);
    config.worker = {
      .address = master_address,
      .port = master_port,
    };
  }

  return config;
}

} // namespace init
} // namespace thd
