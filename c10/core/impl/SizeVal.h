#pragma once


namespace c10 {
namespace impl {


class C10_API AbstractSizeVal;


class C10_API SizeVal {
public:
    operator int64_t*() {
        return &field_.val;
    }

    operator const int64_t*() const {
        return &field_.val;
    }

    operator int64_t() {
        return field_.val;
    }

    operator int64_t() const {
        return field_.val;
    }

    SizeVal& operator=(int64_t val) {
      this->field_.val = val;
      return *this;
    }

    bool operator<(SizeVal other) {
        return (int64_t)this < (int64_t)other;
    }

private:
    union {
    int64_t val;
    AbstractSizeVal* ptr;
    } field_;
};


static inline std::ostream& operator<<(std::ostream& out, const SizeVal& obj) {
    out << (int64_t)obj;
    return out;
}

}} 