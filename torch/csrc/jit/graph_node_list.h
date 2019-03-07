#pragma once

#include <c10/util/Exception.h>

namespace torch {
namespace jit {

// Intrusive doubly linked lists with sane reverse iterators.
// The header file is named generic_graph_node_list.h because it is ONLY
// used for Graph's Node lists, and if you want to use it for other
// things, you will have to do some refactoring.
//
// At the moment, the templated type T must support a few operations:
//
//  - It must have a field: T* next_in_graph[2] = { nullptr, nullptr };
//    which are used for the intrusive linked list pointers.
//
//  - It must have a method 'destroy()', which removes T from the
//    list and frees a T.
//
// In practice, we are only using it with Node and const Node.  'destroy()'
// needs to be renegotiated if you want to use this somewhere else.
//
// Besides the benefits of being intrusive, unlike std::list, these lists handle
// forward and backward iteration uniformly because we require a
// "before-first-element" sentinel.  This means that reverse iterators
// physically point to the element they logically point to, rather than
// the off-by-one behavior for all standard library reverse iterators.

static constexpr int kNextDirection = 0;
static constexpr int kPrevDirection = 1;

template <typename T>
struct generic_graph_node_list;

template <typename T>
struct generic_graph_node_list_iterator;

struct Node;
using graph_node_list = generic_graph_node_list<Node>;
using const_graph_node_list = generic_graph_node_list<const Node>;
using graph_node_list_iterator = generic_graph_node_list_iterator<Node>;
using const_graph_node_list_iterator =
    generic_graph_node_list_iterator<const Node>;

template <typename T>
struct generic_graph_node_list_iterator {
  generic_graph_node_list_iterator() : cur(nullptr), d(kNextDirection) {}
  generic_graph_node_list_iterator(T* cur, int d) : cur(cur), d(d) {}
  generic_graph_node_list_iterator(
      const generic_graph_node_list_iterator& rhs) = default;
  generic_graph_node_list_iterator(generic_graph_node_list_iterator&& rhs) =
      default;
  generic_graph_node_list_iterator& operator=(
      const generic_graph_node_list_iterator& rhs) = default;
  generic_graph_node_list_iterator& operator=(
      generic_graph_node_list_iterator&& rhs) = default;
  T* operator*() const {
    return cur;
  }
  T* operator->() const {
    return cur;
  }
  generic_graph_node_list_iterator& operator++() {
    AT_ASSERT(cur);
    cur = cur->next_in_graph[d];
    return *this;
  }
  generic_graph_node_list_iterator operator++(int) {
    generic_graph_node_list_iterator old = *this;
    ++(*this);
    return old;
  }
  generic_graph_node_list_iterator& operator--() {
    AT_ASSERT(cur);
    cur = cur->next_in_graph[reverseDir()];
    return *this;
  }
  generic_graph_node_list_iterator operator--(int) {
    generic_graph_node_list_iterator old = *this;
    --(*this);
    return old;
  }

  // erase cur without invalidating this iterator
  // named differently from destroy so that ->/. bugs do not
  // silently cause the wrong one to be called.
  // iterator will point to the previous entry after call
  void destroyCurrent() {
    T* n = cur;
    cur = cur->next_in_graph[reverseDir()];
    n->destroy();
  }
  generic_graph_node_list_iterator reverse() {
    return generic_graph_node_list_iterator(cur, reverseDir());
  }

 private:
  int reverseDir() {
    return d == kNextDirection ? kPrevDirection : kNextDirection;
  }
  T* cur;
  int d; // direction 0 is forward 1 is reverse, see next_in_graph
};

template <typename T>
struct generic_graph_node_list {
  using iterator = generic_graph_node_list_iterator<T>;
  using const_iterator = generic_graph_node_list_iterator<const T>;
  generic_graph_node_list_iterator<T> begin() {
    return generic_graph_node_list_iterator<T>(head->next_in_graph[d], d);
  }
  generic_graph_node_list_iterator<const T> begin() const {
    return generic_graph_node_list_iterator<const T>(head->next_in_graph[d], d);
  }
  generic_graph_node_list_iterator<T> end() {
    return generic_graph_node_list_iterator<T>(head, d);
  }
  generic_graph_node_list_iterator<const T> end() const {
    return generic_graph_node_list_iterator<const T>(head, d);
  }
  generic_graph_node_list_iterator<T> rbegin() {
    return reverse().begin();
  }
  generic_graph_node_list_iterator<const T> rbegin() const {
    return reverse().begin();
  }
  generic_graph_node_list_iterator<T> rend() {
    return reverse().end();
  }
  generic_graph_node_list_iterator<const T> rend() const {
    return reverse().end();
  }
  generic_graph_node_list reverse() {
    return generic_graph_node_list(
        head, d == kNextDirection ? kPrevDirection : kNextDirection);
  }
  const generic_graph_node_list reverse() const {
    return generic_graph_node_list(
        head, d == kNextDirection ? kPrevDirection : kNextDirection);
  }
  T* front() {
    return head->next_in_graph[d];
  }
  const T* front() const {
    return head->next_in_graph[d];
  }
  T* back() {
    return head->next_in_graph[!d];
  }
  const T* back() const {
    return head->next_in_graph[!d];
  }
  generic_graph_node_list(T* head, int d) : head(head), d(d) {}

 private:
  T* head;
  int d;
};

template <typename T>
static inline bool operator==(
    generic_graph_node_list_iterator<T> a,
    generic_graph_node_list_iterator<T> b) {
  return *a == *b;
}

template <typename T>
static inline bool operator!=(
    generic_graph_node_list_iterator<T> a,
    generic_graph_node_list_iterator<T> b) {
  return *a != *b;
}

} // namespace jit
} // namespace torch

namespace std {

template <typename T>
struct iterator_traits<torch::jit::generic_graph_node_list_iterator<T>> {
  using difference_type = int64_t;
  using value_type = T*;
  using pointer = T**;
  using reference = T*&;
  using iterator_category = bidirectional_iterator_tag;
};

} // namespace std
