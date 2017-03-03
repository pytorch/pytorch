# Gloo documentation

Documentation is split by domain. This file contains a general
overview of these domains and how they interact.

## Index

* [Overview](readme.md) -- this file

* [Algorithms](algorithms.md) -- index of algorithms and their
  semantics

* [Transport details](transport.md) -- the transport API and its
  implementations

* [CUDA integration](cuda.md) -- integration of CUDA aware Gloo
  algorithms with existing CUDA code

* [Latency optimization](latency.md) -- series of tips and tricks to
  improve algorithm latency

## Overview

Gloo algorithms are collective algoritms, meaning they can run in
parallel across two or more processes/machines. To be able to execute
across multiple machines, they first need to find each other. We call
this _rendezvous_ and it is the first thing to address when
integrating Gloo into your code base.

Once rendezvous completes, participating machines have setup
connections to one another, either in a full mesh (every machine has a
bidirectional communication channel to every other machine), or some
subset. The required connectivity between machines depends on the type
of algorithm that is used. For example, a ring algorithm only needs
communication channels to a machine's neighbors.

Every participating process knows about the number of participating
processes, and its _rank_ (or 0-based index) within the list of
participating processes. This state, as well as the state needed to
store the persistent communication channels, is stored in a
`gloo::Context` class. Gloo does not maintain global state or
thread-local state. This means that you can setup as many contexts as
needed, and introduce as much parallelism as needed by your
application.

## Rendezvous

The rendezvous process needs to happen exactly once per Gloo context.
It makes participating Gloo processes exchange details for setting up
their communication channels. For example, when the TCP transport is
used, processes exchange IP address and port number details of
listening sockets.

Rendezvous is abstracted as a key/value interface to a store that is
accessible by all participating processes. Every process is
responsible for setting a number of keys and will wait until their
peers have set their keys. The values stored against these keys hold
the information that is passed to the transport layer.

This interface is defined in [`store.h`](../gloo/rendezvous/store.h).

### HashStore

The [HashStore](../gloo/rendezvous/hash_store.cc) is an in-process
implementation of this interface. This is realistically not useful in
any application but integration tests.

### RedisStore

The [RedisStore](../gloo/rendezvous/redis_store.cc) implementation uses
the Hiredis library to set/get values against a Redis server. This
server needs to be accessible to all participating machines.

Since the keys used by the Redis implementation are accessible to any
process using that server -- which would prevent usage for concurrent
rendezvous executation -- the
[PrefixStore](../gloo/rendezvous/prefix_store.cc) can be used to scope
rendezvous to a particular namespace.

### ...

Any class that inherits from the `gloo::rendezvous::Store` abstract
base class can be used for rendezvous.

## Anything else?

If you find particular documentation is missing, please consider
[contributing](../CONTRIBUTING.md).
