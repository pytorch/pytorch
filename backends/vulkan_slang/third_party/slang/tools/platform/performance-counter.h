#ifndef PLATFORM_PERFORMANCE_COUNTER_H
#define PLATFORM_PERFORMANCE_COUNTER_H

#include <chrono>

namespace platform
{
typedef std::chrono::high_resolution_clock::time_point TimePoint;
typedef std::chrono::high_resolution_clock::duration Duration;

class PerformanceCounter
{
public:
    static inline TimePoint now() { return std::chrono::high_resolution_clock::now(); }
    static inline Duration getElapsedTime(TimePoint counter) { return now() - counter; }
    static inline float getElapsedTimeInSeconds(TimePoint counter)
    {
        return (float)toSeconds(now() - counter);
    }
    static inline double toSeconds(Duration duration)
    {
        return std::chrono::duration<float>(duration).count();
    }
};
} // namespace platform

#endif
