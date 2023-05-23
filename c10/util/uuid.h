#include <random>
#include <sstream>

namespace c10 {

namespace uuid {

static std::string generate_uuid_v4() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_int_distribution<> dis_hex(0, 15);
  static std::uniform_int_distribution<> dis_89ab(8, 11);

  std::stringstream uuid;
  int i;
  uuid << std::hex;
  for (i = 0; i < 8; i++) {
    uuid << dis_hex(gen);
  }
  uuid << "-";
  for (i = 0; i < 4; i++) {
    uuid << dis_hex(gen);
  }
  // set the four most significant bits of the 7th byte to 0100'B, so the high
  // nibble is "4"
  uuid << "-4";
  for (i = 0; i < 3; i++) {
    uuid << dis_hex(gen);
  }
  uuid << "-";
  // set the two most significant bits of the 9th byte to 10'B, so the high
  // nibble will be one of "8", "9", "A", or "B"0
  uuid << dis_89ab(gen);
  for (i = 0; i < 3; i++) {
    uuid << dis_hex(gen);
  }
  uuid << "-";
  for (i = 0; i < 12; i++) {
    uuid << dis_hex(gen);
  };
  return uuid.str();
}
} // namespace uuid
} // namespace c10
