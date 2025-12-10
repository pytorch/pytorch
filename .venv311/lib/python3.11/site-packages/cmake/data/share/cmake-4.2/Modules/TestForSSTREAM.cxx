#include <sstream>
int main(int, char*[])
{
  std::ostringstream os;
  os << "12345";
  if (os.str().size() == 5) {
    return 0;
  }
  return -1;
}
