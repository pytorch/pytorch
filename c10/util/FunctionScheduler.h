#pragma once

#include <functional>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <set>

namespace c10 {

class Job {
    int _id;
    std::function<void()> _function;
    std::chrono::microseconds _interval;
    std::chrono::time_point<std::chrono::steady_clock> _next_run;

public:
    static inline bool lt(Job a, Job b) { return a.next_run() < b.next_run(); }

    Job(std::function<void()> function, std::chrono::microseconds interval);

    std::chrono::time_point<std::chrono::steady_clock> next_run() const;
    std::chrono::microseconds interval() const;
    int id() const;

    void set_next_run(std::chrono::time_point<std::chrono::steady_clock> next_run);
    void set_id(int id);

    void run();
};

class FunctionScheduler {
    int _current_id = 0;
    std::atomic_bool _running = false;
    std::multiset<Job, decltype(&Job::lt)> _queue;
    std::vector<Job> _jobs;
    std::thread _thread;

    void runNextJob();
    std::chrono::microseconds getNextInterval() const;
    int id();

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
