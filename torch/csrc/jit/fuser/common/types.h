// #pragma once

// #include <cassert>
// #include <iostream>

// namespace torch {
// namespace jit {
// namespace fuser {

// //Order based on type promotion rules
// enum class CType {
//   kFloat32,
//   kInt32,
//   kStatement,
//   kNull
// };

// // Data types for scalar and vector elements.
// class DType {
//  public:
//   DType(CType ctype, int lanes = 1)
//       : ctype_(ctype), lanes_(lanes) {}

//   DType(const DType &d2):ctype_(d2.ctype()), lanes_(d2.lanes()){}

//   const CType& ctype() const{
//     return ctype_;
//   }

//   const int& lanes() const {
//     return lanes_;
//   }

//   bool operator==(const DType& other) const {
//     return ctype_ == other.ctype_ && lanes_ == other.lanes_;
//   }
//   bool operator!=(const DType& other) const {
//     return !(*this == other);
//   }

//  private:
//   const CType ctype_;
//   const int lanes_; // the width of the element for a vector time
// };

// std::ostream& operator<<(std::ostream& os, const DType& dtype);

// bool is_scalar(const CType& type);

// CType promote(const CType& t1, const CType& t2);

// bool is_scalar(const DType& type);

// DType promote(const DType& t1, const DType& t2);

// } // namespace fuser
// } // namespace jit
// } // namespace torch
