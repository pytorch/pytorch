namespace c10 {

struct in_place_t { explicit in_place_t() = default; };

constexpr in_place_t in_place{};

} // namespace c10
