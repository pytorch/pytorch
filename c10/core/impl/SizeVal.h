#pragma once
#include <iostream>

namespace c10 {
namespace impl {


class C10_API AbstractSizeVal {
    public:
    AbstractSizeVal(int64_t val): value(val) {}

    int64_t* get() {
        std::cout << "Accessing abstract size: " << value <<std::endl;
        return &value;
    }

    const int64_t* get() const {
        std::cout << "Accessing const abstract size: " << value <<std::endl;
        return &value;
    }

    AbstractSizeVal* operator+(int64_t other) const {
        std::cout << "Adding const abstract size: " << value <<std::endl;
        AbstractSizeVal* res = new AbstractSizeVal(value + other);
        return res;
    }

    AbstractSizeVal* operator+(int64_t other) {
        std::cout << "Adding abstract size: " << value <<std::endl;
        AbstractSizeVal* res = new AbstractSizeVal(value + other);
        return res;
    }


    int64_t value;
};

class C10_API SizeVal {
public:
    SizeVal() = default;

    SizeVal(int inp) {
        this->field_.val = inp;
    }

    SizeVal(AbstractSizeVal* ptr) {
        int64_t saved_ptr = (int64_t)ptr;
        saved_ptr |= ((int64_t)1 << 63);
        this->field_.ptr = (AbstractSizeVal*) saved_ptr;
    }

    SizeVal(const int64_t& inp) {
        this->field_.val = inp;
    }

    SizeVal(int64_t&& inp) {
        this->field_.val = inp;
    }

    operator int64_t*() {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return ((AbstractSizeVal*)real_ptr)->get();
        } else {
            return &field_.val;
        }
    }
 
    operator const int64_t*() const {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return ((AbstractSizeVal*)real_ptr)->get();
        } else {
            return &field_.val;
        }
    }

    operator int64_t() {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return *((AbstractSizeVal*)real_ptr)->get();
        } else {
            return field_.val;
        }
    }

    operator int64_t() const {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return *((AbstractSizeVal*)real_ptr)->get();
        } else {
            return field_.val;
        }
    }

    SizeVal operator-(int64_t other) const {
        return SizeVal(field_.val - other);
    }

    SizeVal operator-(int other) const {
        return SizeVal(field_.val - other);
    }

    SizeVal operator+(int64_t other) const {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return SizeVal(*((AbstractSizeVal*)real_ptr) + other);
        } else {
            return SizeVal(field_.val + other);
        }
    }

    SizeVal operator+(const SizeVal& other) const {
        return SizeVal(field_.val + other.field_.val);
    }

    SizeVal operator+(int64_t other) {
        if (field_.val < 0) {
            int64_t real_ptr = (int64_t)field_.ptr;
            real_ptr &= ~((int64_t)1 << 63);
            return SizeVal(*((AbstractSizeVal*)real_ptr) + other);
        } else {
            return SizeVal(field_.val + other);
        }
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

    SizeVal& operator=(const long unsigned int val) {
        this->field_.val = val;
        return *this;
    }

    SizeVal& operator=(const int val) {
        this->field_.val = val;
        return *this;
    }

    SizeVal& operator=(AbstractSizeVal* ptr) {
        int64_t saved_ptr = (int64_t)ptr;
        saved_ptr |= ((int64_t)1 << 63);
        this->field_.ptr = (AbstractSizeVal*) saved_ptr;
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