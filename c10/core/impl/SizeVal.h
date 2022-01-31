#pragma once


namespace c10 {
namespace impl {


class C10_API AbstractSizeVal;


class C10_API SizeVal {
public:
    SizeVal() = default;

    SizeVal(int inp) {
        this->field_.val = inp;
    }

    SizeVal(const int64_t& inp) {
        this->field_.val = inp;
    }

    SizeVal(int64_t&& inp) {
        this->field_.val = inp;
    }

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

    SizeVal operator-(int64_t other) const {
        return SizeVal(field_.val - other);
    }

    SizeVal operator-(int other) const {
        return SizeVal(field_.val - other);
    }

    SizeVal operator+(int64_t other) const {
        return SizeVal(field_.val + other);
    }

    SizeVal operator+(const SizeVal& other) const {
        return SizeVal(field_.val + other.field_.val);
    }

    SizeVal operator+(int64_t other) {
        return SizeVal(field_.val + other);
    }

    SizeVal operator+(int other) const {
        return SizeVal(field_.val + other);
    }

    SizeVal operator+(int other) {
        return SizeVal(field_.val + other);
    }

    SizeVal& operator=(int64_t val) {
      this->field_.val = val;
      return *this;
    }

    bool operator==(const SizeVal& other) const {
        return (int64_t)this == (int64_t)other;
    }

    bool operator!=(const SizeVal& other) const {
        return (int64_t)this != (int64_t)other;
    }

    bool operator!=(int64_t other) const {
        return (int64_t)this != (int64_t)other;
    }

    bool operator!=(int other) const {
        return (int64_t)this != (int64_t)other;
    }

    // We need both int and int64_t here otherwise plain numbers from size == 0
    // are confusing for gcc ??
    // Same all around this file
    bool operator==(const int64_t& other) const {
        return (int64_t)this == other;
    }
    bool operator==(const int& other) const {
        return (int64_t)this == other;
    }

    bool operator<(SizeVal other) const {
        return (int64_t)this < (int64_t)other;
    }

    bool operator<(const int64_t& other) const {
        return (int64_t)this < other;
    }
    bool operator<(const int& other) const {
        return (int64_t)this < other;
    }

    bool operator>(SizeVal other) const {
        return (int64_t)this > (int64_t)other;
    }

    bool operator>(const int64_t& other) const {
        return (int64_t)this > other;
    }
    bool operator>(const int& other) const {
        return (int64_t)this > other;
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

inline SizeVal operator+(const int64_t& first, const SizeVal& second) {
    return second + first;
}

inline SizeVal operator+(const int& first, const SizeVal& second) {
    return second + first;
}

// c10::impl::size_val_vec_to_int(
static inline std::vector<int64_t> size_val_vec_to_int(const std::vector<SizeVal>& inp) {
    return std::vector<int64_t>((int64_t*)inp.data(), (int64_t*)inp.data() + inp.size());
}


}} 