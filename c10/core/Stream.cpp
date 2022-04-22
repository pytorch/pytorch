#include <c10/core/Stream.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace c10 {

// Return whether all asynchronous work previously enqueued on this stream
// has completed running on the device.
bool Stream::query() const {
  impl::VirtualGuardImpl impl{device_.type()};
  return impl.queryStream(*this);
}

// Wait (by blocking the calling thread) until all asynchronous work enqueued
// on this stream has completed running on the device.
void Stream::synchronize() const {
  impl::VirtualGuardImpl impl{device_.type()};
  impl.synchronizeStream(*this);
}

// Not very parsable, but I don't know a good compact syntax for streams.
// Feel free to change this into something more compact if needed.
std::ostream& operator<<(std::ostream& stream, const Stream& s) {
  stream << "stream " << s.id() << " on device " << s.device();
  return stream;
}

} // namespace c10
