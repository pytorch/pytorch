#include <THD/base/init_methods/InitMethod.hpp>
#include <THD/base/init_methods/InitMethodUtils.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <system_error>
#include <thread>

namespace thd {
namespace init {

namespace {

void lockLoop(int fd, struct flock& oflock) {
  while (true) {
    int err = ::fcntl(fd, F_SETLKW, &oflock);
    if (err == 0)
      break;
    else if (errno == EINTR)
      continue;
    else
      throw std::system_error(errno, std::system_category());
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

// file_descriptor, number_of_lines_in_file
std::pair<int, size_t> waitForGroup(
    std::string file_path,
    std::string group_name,
    std::fstream& file) {
  int fd;
  std::string content;
  struct stat fd_stat, path_stat;
  // Loop until the file is either empty, or filled with ours group_name
  while (true) {
    // Loop until we have an open, locked and valid file
    while (true) {
      fd = ::open(file_path.c_str(), O_RDWR | O_CREAT, 0644);
      if (fd == -1) {
        throw std::system_error(
            fd,
            std::generic_category(),
            "cannot access '" + file_path + "' file");
      }
      lockFile(fd);

      // This helps prevent a race when while we were waiting for the lock,
      // the file has been removed from the fs
      SYSCHECK(::fstat(fd, &fd_stat));
      int err = stat(file_path.c_str(), &path_stat);
      if (err == 0 && fd_stat.st_dev == path_stat.st_dev &&
          fd_stat.st_ino == path_stat.st_ino) {
        break;
      }
      ::close(fd);
    }

    file.close();
    file.open(file_path);
    content = {std::istreambuf_iterator<char>(file),
               std::istreambuf_iterator<char>()};

    if (content.length() == 0 || content.find(group_name) == 0)
      break;

    unlockFile(fd);
    ::close(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  return {fd, std::count(content.begin(), content.end(), '\n')};
}

size_t waitForData(int fd, std::fstream& file, rank_type world_size) {
  size_t lines = 0;
  // Wait until all processes will write their info
  while (lines < world_size) {
    unlockFile(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    lockFile(fd);

    file.seekp(0, std::ios_base::beg);
    file.sync();
    std::string content = {std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>()};
    lines = std::count(content.begin(), content.end(), '\n');
  }

  file.seekp(0, std::ios_base::beg);
  return lines;
}

// master_port, master_addrs, ranks
std::tuple<port_type, std::vector<std::string>, std::vector<int>> parseFile(
    std::fstream& file,
    rank_type world_size,
    std::string group_name) {
  port_type master_port;
  std::vector<std::string> master_addrs;
  std::vector<int> ranks(world_size);
  // Parse the file
  for (size_t i = 0; i < world_size; ++i) {
    std::string proc_group_name;
    size_t proc_addrs_count;
    int proc_rank;
    port_type proc_port;

    file >> proc_group_name >> proc_rank >> proc_port >> proc_addrs_count;
    if (proc_group_name != group_name) {
      throw std::logic_error("proc_group_name != group_name");
    }

    std::vector<std::string> proc_addrs(proc_addrs_count);
    for (auto& str : proc_addrs) {
      file >> str;
    }

    ranks[i] = proc_rank;
    /*
     * Master data is found only when:
     *  1. proc_rank has been manually assigned as 0 (first condition)
     *  2. process has no assigned rank, and it hasn't been initialized yet.
     */
    if (proc_rank == 0 || (proc_rank == -1 && master_addrs.size() == 0)) {
      master_port = proc_port;
      master_addrs = std::move(proc_addrs);
    }
  }

  // Ensure there are no duplicates
  for (size_t i = 0; i < ranks.size(); ++i) {
    for (size_t j = i + 1; j < ranks.size(); ++j) {
      if (ranks[i] >= 0 && (ranks[i] == ranks[j]))
        throw std::logic_error("more than one node have assigned same rank");
    }
  }

  return std::make_tuple(master_port, master_addrs, ranks);
}

} // anonymous namespace

InitMethod::Config initFile(
    std::string argument,
    int world_size_r,
    std::string group_name,
    int assigned_rank) {
  group_name.append("#"); // To make sure it's not empty
  std::string file_path = argument.substr(7); // chop "file://"
  rank_type world_size;
  try {
    world_size = convertToRank(world_size_r);
  } catch (std::exception& e) {
    if (world_size_r == -1) {
      throw std::invalid_argument(
          "world_size is not set - it is required for "
          "`file://` init methods with this backend");
    }
    throw std::invalid_argument("invalid world_size");
  }

  InitMethod::Config config;
  int fd;
  size_t order;
  std::fstream file;

  std::tie(fd, order) = waitForGroup(file_path, group_name, file);
  // NOTE: the function returns a locked fd

  int listen_socket;
  port_type port;
  std::tie(listen_socket, port) = listen();

  // Append our information
  auto if_addrs = getInterfaceAddresses();
  file << group_name << ' ' << assigned_rank << ' ' << port << ' '
       << if_addrs.size();
  for (auto addr_str : if_addrs) {
    file << ' ' << addr_str;
  }
  file << std::endl;

  size_t lines = waitForData(fd, file, world_size);

  port_type master_port = -1;
  std::vector<std::string> master_addrs;
  std::vector<int> ranks;
  std::tie(master_port, master_addrs, ranks) =
      parseFile(file, world_size, group_name);

  config.rank = getRank(ranks, assigned_rank, order);

  // Last process removes the file.
  file.seekp(0, std::ios_base::end);
  file << std::endl;
  lines++;
  if (lines == 2 * world_size) {
    ::remove(file_path.c_str());
  }

  file.close();
  unlockFile(fd);

  config.world_size = world_size;
  if (config.rank == 0) {
    config.public_address = discoverWorkers(listen_socket, world_size);
    config.master = {
        .listen_socket = listen_socket,
        .listen_port = master_port,
    };
  } else {
    ::close(listen_socket);

    std::string master_address;
    std::tie(master_address, config.public_address) =
        discoverMaster(master_addrs, master_port);
    config.worker = {
        .master_addr = master_address,
        .master_port = master_port,
    };
  }

  return config;
}

} // namespace init
} // namespace thd
