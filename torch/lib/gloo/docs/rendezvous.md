# Rendezvous

The rendezvous process needs to happen exactly once per Gloo context.
It makes participating Gloo processes exchange details for setting up
their communication channels. For example, when the TCP transport is
used, processes exchange IP address and port number details of
listening sockets.

For example:

```c++
// Initialize context
auto myRank = 0;  // Rank of this process within list of participating processes
auto contextSize = 2;  // Number of participating processes
gloo::rendezvous::Context context(myRank, contextSize);

// Perform rendezvous for TCP pairs
auto dev = gloo::transport::tcp::CreateDevice();
gloo::rendezvous::RedisStore redis("redishost");
context.connectFullMesh(redis, dev);
```

## Using a key/value store

Rendezvous can be executed by accessing a key/value store that is
accessible by all participating processes. Every process is
responsible for setting a number of keys and will wait until their
peers have set their keys. The values stored against these keys hold
the information that is passed to the transport layer.

This interface is defined in [`store.h`](../gloo/rendezvous/store.h).

### HashStore

The [HashStore](../gloo/rendezvous/hash_store.cc) is an in-process
implementation of this interface. This is realistically not useful in
any application but integration tests.

### FileStore

The [FileStore](../gloo/rendezvous/file_store.cc) is a simple file system based
implementation of this interface. The primary use case is multi-process testing,
but it may be useful in other scenarios with a shared file system.

### RedisStore

The [RedisStore](../gloo/rendezvous/redis_store.cc) implementation uses
the Hiredis library to set/get values against a Redis server. This
server needs to be accessible to all participating machines.

Since the keys used by the Redis implementation are accessible to any
process using that server -- which would prevent usage for concurrent
rendezvous execution -- the
[PrefixStore](../gloo/rendezvous/prefix_store.cc) can be used to scope
rendezvous to a particular namespace.

### YourCustomStore

There are many more key/value stores that can be used for rendezvous
(e.g. [etcd](https://coreos.com/etcd) or [ZooKeeper](https://zookeeper.apache.org/)).
As long as a C or C++ interface for your store of choice is available,
is relatively easy to hook it up to the Gloo rendezvous process.
See the `gloo::rendezvous::Store` abstract base class for the interface to implement.

## Using MPI

If you are already using MPI to run jobs across machines, getting started with
Gloo should be staightforward. Instead of using a separate key/value store for
rendezvous, the existing MPI communicator is used to create contexts.
Make sure to compile Gloo with `USE_MPI=ON`.

Note that Gloo does **NOT** use MPI for anything after a context has been created.
