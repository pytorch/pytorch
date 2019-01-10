#pragma once
#include <cmath>
namespace at
{
namespace native
{
static inline int64_t getSquareSize(int64_t vector_size)
{
  // solve the equation of d * (d - 1) / 2 == vector_size
  // d = (std::sqrt(8 * size + 1) + 1) / 2;
  // d is 1 if vector_size is 0, otherwise
  // grab the closest value to the square root of the number
  // of elements times 2 to see if the number of elements
  // is indeed a binomial coefficient.
  int64_t d = 1;
  if (vector_size > 0)
  {
    d = std::ceil(std::sqrt(vector_size * 2));
  }
  AT_CHECK(d * (d - 1) / 2 == vector_size,
    "Incompatible vector size [", vector_size,
    "]. It must be a binomial coefficient n choose 2 for some integer");
  return d;
}

}  // at::native
}  // at
