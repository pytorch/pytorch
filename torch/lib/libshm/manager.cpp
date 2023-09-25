#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

#include <c10/util/tempfile.h>

#include <libshm/err.h>
#include <libshm/socket.h>

const int SHUTDOWN_TIMEOUT = 2000; // 2s

#ifdef DEBUG_LOG
#define COLOR "\033[31;1m"
#define RESET "\033[0m"
#define __DEBUG(msg, ...) fprintf(stderr, COLOR msg "%c" RESET, __VA_ARGS__);
#define DEBUG(...) __DEBUG(__VA_ARGS__, '\n')
#else
#define DEBUG(...) (void)0
#endif

struct ClientSession {
  ClientSession(ManagerSocket s) : socket(std::move(s)), pid(0) {}

  ManagerSocket socket;
  pid_t pid;
};

std::vector<struct pollfd> pollfds;
std::unordered_map<int, ClientSession> client_sessions;
// TODO: check if objects have been freed from time to time
std::set<std::string> used_objects;

void register_fd(int fd) {
  struct pollfd pfd = {0};
  pfd.fd = fd;
  pfd.events = POLLIN;
  pollfds.push_back(pfd);
}

void unregister_fd(int fd) {
  pollfds.erase(
      std::remove_if(
          pollfds.begin(),
          pollfds.end(),
          [fd](const struct pollfd& pfd) { return pfd.fd == fd; }),
      pollfds.end());
  client_sessions.erase(fd);
}

void print_init_message(const char* message) {
  write(1, message, strlen(message));
  write(1, "\n", 1);
}

bool object_exists(const char* name) {
  int fd = shm_open(name, O_RDONLY, 0);
  if (fd >= 0) {
    close(fd);
    return true;
  } else {
    return false;
  }
}

void free_used_object(const std::string& name) {
  if (!object_exists(name.c_str())) {
    DEBUG("object %s appears to have been freed", name.c_str());
    used_objects.erase(name);
  } else {
    DEBUG("object %s still exists", name.c_str());
  }
}

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char* argv[]) {
  setsid(); // Daemonize the process

  std::unique_ptr<ManagerServerSocket> srv_socket;
  std::optional<c10::TempDir> tempdir;
  try {
    tempdir = c10::try_make_tempdir(/*name_prefix=*/"torch-shm-dir-");
    if (!tempdir.has_value()) {
      throw std::runtime_error(
          "could not generate a random directory for manager socket");
    }

    std::string tempfile = tempdir->name + "/manager.sock";

    srv_socket = std::make_unique<ManagerServerSocket>(tempfile);
    register_fd(srv_socket->socket_fd);
    print_init_message(tempfile.c_str());
    DEBUG("opened socket %s", tempfile.c_str());
  } catch (const std::exception& e) {
    std::string message("ERROR: ");
    message += e.what();
    print_init_message(message.c_str());
    return 1;
  } catch (...) {
    print_init_message("ERROR: unhandled exception");
    return 1;
  }

  int timeout = -1;
  std::vector<int> to_add;
  std::vector<int> to_remove;
  for (;;) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    int nevents;
    if (client_sessions.empty())
      timeout = SHUTDOWN_TIMEOUT;
    SYSCHECK_ERR_RETURN_NEG1(
        nevents = poll(pollfds.data(), pollfds.size(), timeout));
    timeout = -1;
    if (nevents == 0 && client_sessions.empty())
      break;

    for (auto& pfd : pollfds) {
      if (pfd.revents & (POLLERR | POLLHUP)) {
        // some process died
        DEBUG("detaching process");
        auto& session = client_sessions.at(pfd.fd);
        (void)session;
        DEBUG("%d has died", session.pid);
        to_remove.push_back(pfd.fd);
      } else if (pfd.revents & POLLIN) {
        if (pfd.fd == srv_socket->socket_fd) {
          // someone is joining
          DEBUG("registered new client");
          auto client = srv_socket->accept();
          int fd = client.socket_fd;
          to_add.push_back(fd);
          client_sessions.emplace(fd, std::move(client));
        } else {
          // someone wants to register a segment
          DEBUG("got alloc info");
          auto& session = client_sessions.at(pfd.fd);
          AllocInfo info = session.socket.receive();
          session.pid = info.pid;
          DEBUG(
              "got alloc info: %d %d %s",
              (int)info.free,
              info.pid,
              info.filename);
          if (info.free) {
            free_used_object(info.filename);
          } else {
            used_objects.insert(info.filename);
            DEBUG("registered object %s", info.filename);
            session.socket.confirm();
          }
        }
      }
    }

    for (int fd : to_add)
      register_fd(fd);
    to_add.clear();

    for (int fd : to_remove)
      unregister_fd(fd);
    to_remove.clear();
  }

  for (auto& obj_name : used_objects) {
    DEBUG("freeing %s", obj_name.c_str());
    shm_unlink(obj_name.c_str());
  }

  // Clean up file descriptors
  for (auto& pfd : pollfds) {
    unregister_fd(pfd.fd);
  }
  // Clean up manager.sock
  srv_socket->remove();
  // Clean up directory automatically

  DEBUG("manager done");
  return 0;
}
