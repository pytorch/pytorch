#pragma once

#include <functional>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>

namespace c10 {

class Job {
    std::function<void()> _function;
    std::chrono::microseconds _interval;

public:
    Job(std::function<void()> function, std::chrono::microseconds interval);

    std::chrono::microseconds delta() const;
    void run();
};

class FunctionScheduler {
    std::atomic_bool _running;
    std::vector<Job> _jobs;
    std::thread _thread;

    void runNextJob();
    std::chrono::microseconds getNextInterval() const;

public:
    FunctionScheduler();
    ~FunctionScheduler();

    int scheduleJob(Job job);
    int scheduleJob(std::function<void()> function, std::chrono::microseconds interval);

    int removeJob(int id);

    void start();
    void stop();
};

} // namespace c10
