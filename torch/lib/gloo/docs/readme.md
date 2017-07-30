# Gloo documentation

Documentation is split by domain. This file contains a general
overview of these domains and how they interact.

## Index

* [Overview](readme.md) -- this file

* [Rendezvous](rendezvous.md) -- creating a `gloo::Context`
  
* [Algorithms](algorithms.md) -- index of collective algorithms
  and their semantics and complexity

* [Transport details](transport.md) -- the transport API and its
  implementations

* [CUDA integration](cuda.md) -- integration of CUDA aware Gloo
  algorithms with existing CUDA code

* [Latency optimization](latency.md) -- number of tips and tricks to
  improve performance

## Overview

Gloo algorithms are collective algorithms, meaning they can run in
parallel across two or more processes/machines. To be able to execute
across multiple machines, they first need to find each other. We call
this _rendezvous_ and it is the first thing to address when
integrating Gloo into your code base.
See [`rendezvous.md`](./rendezvous.md) for more information.

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

## Anything else?

If you find particular documentation is missing, please consider
[contributing](../CONTRIBUTING.md).
