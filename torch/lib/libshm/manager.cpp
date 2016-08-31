#include <sys/mman.h>
#include <poll.h>
#include <errno.h>
#include <unistd.h>
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include <unordered_map>

#include "err.h"
#include "socket.h"

const int SHUTDOWN_TIMEOUT = 10000;

#ifdef DEBUG_LOG
#define COLOR "\033[31;1m"
#define RESET "\033[0m"
#define __DEBUG(msg, ...) fprintf(stderr, COLOR msg "%c" RESET, __VA_ARGS__);
#define DEBUG(...) __DEBUG(__VA_ARGS__, '\n')
#else
#define DEBUG(...) (void)0
#endif

struct ClientSession {
  ClientSession(ManagerSocket s): socket(std::move(s)), pid(0) {}

  pid_t pid;
  std::set<std::string> used_objects;
  ManagerSocket socket;
};


std::vector<struct pollfd> pollfds;
std::unordered_map<int, ClientSession> client_sessions;
std::unordered_map<std::string, int> object_refcounts;


void register_fd(int fd) {
  struct pollfd pfd = {0};
  pfd.fd = fd;
  pfd.events = POLLIN;
  pollfds.push_back(pfd);
}


void unregister(int fd) {
  pollfds.erase(
    std::remove_if(pollfds.begin(), pollfds.end(),
        [fd](const struct pollfd &pfd) { return pfd.fd == fd; }),
    pollfds.end());
  client_sessions.erase(fd);
}


void print_init_message(const char *message) {
  write(1, message, strlen(message));
  write(1, "\n", 1);
}

bool object_decref(const std::string &obj_name, bool should_unlink=false) {
  int new_refcount = --object_refcounts[obj_name];
    DEBUG("decreased %s refcount to %d", obj_name.c_str(), new_refcount);
  if (new_refcount == 0) {
    object_refcounts.erase(obj_name);
    if(should_unlink) {
      DEBUG("unlinking %s", obj_name.c_str());
      shm_unlink(obj_name.c_str());
    }
    return true;
  }
  return false;
}


int main(int argc, char *argv[]) {
  setsid();  // Daemonize the process

  std::unique_ptr<ManagerServerSocket> srv_socket;
  try {
    char tmpfile[L_tmpnam];
    if (std::tmpnam(tmpfile) == NULL)
      throw std::exception();
    // TODO: better strategy for generating tmp names
    // TODO: retry on collisions - this can easily fail
    srv_socket.reset(new ManagerServerSocket(std::string(tmpfile)));
    register_fd(srv_socket->socket_fd);
    print_init_message(tmpfile);
    DEBUG("opened socket %s", tmpfile);
  } catch(...) {
    print_init_message("ERROR");
    throw;
  }

  int timeout = -1;
  std::vector<int> to_add;
  std::vector<int> to_remove;
  for (;;) {
    int nevents;
    if (client_sessions.size() == 0)
      timeout = SHUTDOWN_TIMEOUT;
    SYSCHECK(nevents = poll(pollfds.data(), pollfds.size(), timeout));
    timeout = -1;
    if (nevents == 0 && client_sessions.size() == 0)
      break;

    for (struct pollfd &pfd: pollfds) {
      if (pfd.revents & (POLLERR | POLLHUP)) {
        // some process died
        DEBUG("detaching process");
        auto &session = client_sessions.at(pfd.fd);
        DEBUG("%d has died", session.pid);
        for (auto &obj_name: session.used_objects)
            object_decref(obj_name, true);
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
          auto &session = client_sessions.at(pfd.fd);
          AllocInfo info = session.socket.recieve();
          session.pid = info.pid;
          DEBUG("got alloc info: %d %d %s", (int)info.free, info.pid, info.filename);
          if (info.free) {
            session.used_objects.erase(info.filename);
            object_decref(info.filename);
          } else {
            object_refcounts[info.filename]++;
            session.used_objects.emplace(info.filename);
            DEBUG("increased %s refcount to %d", info.filename, object_refcounts[info.filename]);
          }
          session.socket.confirm();
        }
      }
    }

    for (int fd: to_add)
    register_fd(fd);
    to_add.clear();

    for (int fd: to_remove)
    unregister(fd);
    to_remove.clear();
  }

  DEBUG("manager done");
  return 0;
}
