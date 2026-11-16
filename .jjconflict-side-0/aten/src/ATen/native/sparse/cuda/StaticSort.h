#pragma once
#include <cutlass/cutlass.h>

/**
 * A Functor class to create a sort for fixed sized arrays/containers with a
 * compile time generated Bose-Nelson sorting network.
 * \tparam NumElements  The number of elements in the array or container to
 * sort. \tparam T            The element type. \tparam Compare      A
 * comparator functor class that returns true if lhs < rhs.
 */
template <unsigned NumElements>
class StaticSort {
  template <class A>
  struct Swap {
    template <class T>
    CUTLASS_HOST_DEVICE void s(T& v0, T& v1) {
      // Explicitly code out the Min and Max to nudge the compiler
      // to generate branchless code.
      T t = v0 < v1 ? v0 : v1; // Min
      v1 = v0 < v1 ? v1 : v0; // Max
      v0 = t;
    }

    CUTLASS_HOST_DEVICE Swap(A& a, const int& i0, const int& i1) {
      s(a[i0], a[i1]);
    }
  };

  template <class A, int I, int J, int X, int Y>
  struct PB {
    CUTLASS_HOST_DEVICE PB(A& a) {
      enum {
        L = X >> 1,
        M = (X & 1 ? Y : Y + 1) >> 1,
        IAddL = I + L,
        XSubL = X - L
      };
      PB<A, I, J, L, M> p0(a);
      PB<A, IAddL, J + M, XSubL, Y - M> p1(a);
      PB<A, IAddL, J, XSubL, M> p2(a);
    }
  };

  template <class A, int I, int J>
  struct PB<A, I, J, 1, 1> {
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s(a, I - 1, J - 1);
    }
  };

  template <class A, int I, int J>
  struct PB<A, I, J, 1, 2> {
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s0(a, I - 1, J);
      Swap<A> s1(a, I - 1, J - 1);
    }
  };

  template <class A, int I, int J>
  struct PB<A, I, J, 2, 1> {
    CUTLASS_HOST_DEVICE PB(A& a) {
      Swap<A> s0(a, I - 1, J - 1);
      Swap<A> s1(a, I, J - 1);
    }
  };

  template <class A, int I, int M, bool Stop = false>
  struct PS {
    CUTLASS_HOST_DEVICE PS(A& a) {
      enum { L = M >> 1, IAddL = I + L, MSubL = M - L };
      PS<A, I, L, (L <= 1)> ps0(a);
      PS<A, IAddL, MSubL, (MSubL <= 1)> ps1(a);
      PB<A, I, IAddL, L, MSubL> pb(a);
    }
  };

  template <class A, int I, int M>
  struct PS<A, I, M, true> {
    CUTLASS_HOST_DEVICE PS(A& a) {}
  };

 public:
  /**
   * Sorts the array/container arr.
   * \param  arr  The array/container to be sorted.
   */
  template <class Container>
  CUTLASS_HOST_DEVICE void operator()(Container& arr) const {
    PS<Container, 1, NumElements, (NumElements <= 1)> ps(arr);
  };

  /**
   * Sorts the array arr.
   * \param  arr  The array to be sorted.
   */
  template <class T>
  CUTLASS_HOST_DEVICE void operator()(T* arr) const {
    PS<T*, 1, NumElements, (NumElements <= 1)> ps(arr);
  };
};
