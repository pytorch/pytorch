#include <c10/util/FunctionScheduler.h>

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

TEST(Job, Initialization) {
  std::function<void()> function = []() {};
  std::chrono::microseconds interval(10);

  c10::Job j(function, interval);

  ASSERT_EQ(j.interval(), interval);
}

TEST(Job, Run) {
  bool ran = false;
  std::function<void()> function = [&ran]() { ran = true; };
  std::chrono::microseconds interval(10);

  c10::Job j(function, interval);
  j.run();

  ASSERT_TRUE(ran);
}

TEST(Run, Initialization) {
  int job_id = 1;
  auto time = std::chrono::steady_clock::now();

  c10::Run r(job_id, time);

  ASSERT_EQ(r.job_id(), job_id);
  ASSERT_EQ(r.time(), time);
}

TEST(Run, gt) {
  int job_id1 = 1;
  int job_id2 = 2;
  auto time1 = std::chrono::steady_clock::now();
  auto time2 = time1 + std::chrono::milliseconds(10);

  auto r1 = std::make_shared<c10::Run>(job_id1, time1);
  auto r2 = std::make_shared<c10::Run>(job_id2, time2);

  ASSERT_TRUE(c10::Run::gt(r2, r1));
  ASSERT_FALSE(c10::Run::gt(r1, r2));
}

TEST(FunctionScheduler, Initialization) {
  c10::FunctionScheduler fs;

  ASSERT_FALSE(fs.isRunning());
  ASSERT_EQ(fs.currentId(), 0);
}

TEST(FunctionScheduler, ScheduleJob) {
  std::function<void()> function = []() {};
  std::chrono::seconds interval(10);

  c10::FunctionScheduler fs;
  int job_id = fs.scheduleJob(function, interval);

  ASSERT_EQ(job_id, 0);
  ASSERT_EQ(fs.currentId(), 1);
}

TEST(FunctionScheduler, RemoveJob) {
  std::function<void()> function = []() {};
  std::chrono::seconds interval(10);

  c10::FunctionScheduler fs;

  int job_id = fs.scheduleJob(function, interval);
  ASSERT_EQ(job_id, 0);
  int remove_id = fs.removeJob(job_id);
  ASSERT_EQ(job_id, remove_id);
}

TEST(FunctionScheduler, RemoveNonExistentJob) {
  c10::FunctionScheduler fs;

  int remove_id = fs.removeJob(0);
  ASSERT_EQ(remove_id, -1);
}

TEST(FunctionScheduler, StartAndStop) {
  c10::FunctionScheduler fs;

  fs.start();
  ASSERT_TRUE(fs.isRunning());

  fs.stop();
  ASSERT_FALSE(fs.isRunning());
}

TEST(FunctionScheduler, RunJobWithZeroInterval) {
  bool ran = false;
  std::function<void()> function = [&ran]() { ran = true; };
  std::chrono::seconds interval(0);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  fs.stop();

  ASSERT_TRUE(ran);
}

TEST(FunctionScheduler, RunJobWithInterval) {
  bool ran = false;
  std::function<void()> function = [&ran]() { ran = true; };
  std::chrono::milliseconds interval(10);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  ASSERT_FALSE(ran);

  std::this_thread::sleep_for(std::chrono::milliseconds(12));
  fs.stop();

  ASSERT_TRUE(ran);
}

TEST(FunctionScheduler, RemoveJobWhileRunning) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(2);

  c10::FunctionScheduler fs;
  int job_id = fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(11));
  fs.removeJob(job_id);
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  fs.stop();

  ASSERT_EQ(counter, 5);
}

TEST(FunctionScheduler, RunMultipleJobs) {
  std::atomic<int> counter = 0;
  std::function<void()> function1 = [&counter]() { counter++; };
  std::function<void()> function2 = [&counter]() { counter += 2; };
  std::chrono::milliseconds interval1(2); // 2ms, 4ms
  std::chrono::milliseconds interval2(4); // 4ms

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  fs.stop();

  ASSERT_EQ(counter, 4);
}

TEST(FunctionScheduler, RunMultipleJobsWithSameInterval) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::milliseconds interval1(2); // 2ms
  std::chrono::milliseconds interval2(2); // 2ms

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  fs.stop();

  ASSERT_EQ(counter1, 1);
  ASSERT_EQ(counter1, 1);
}

TEST(FunctionScheduler, ScheduleJobAfterCurrentlyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::milliseconds interval1(5); // 5ms
  std::chrono::milliseconds interval2(4); // 3+4ms

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 3ms
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 6ms
  ASSERT_TRUE(yes1);
  ASSERT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::milliseconds(2)); // 8ms
  ASSERT_TRUE(yes1);
  ASSERT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeCurrenltyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::milliseconds interval1(10); // 10ms
  std::chrono::milliseconds interval2(2); // 3+2ms

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 3ms
  fs.scheduleJob(function2, interval2);
  ASSERT_FALSE(yes1);
  ASSERT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 6ms
  ASSERT_FALSE(yes1);
  ASSERT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeAndAfterCurrenltyWaiting) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::milliseconds interval1(6); // 6ms
  std::chrono::milliseconds interval2(2); // 3+2ms, 3+4ms

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(3)); // 3ms
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::milliseconds(5)); // 8ms
  ASSERT_EQ(counter1, 1);
  ASSERT_EQ(counter2, 2);
  fs.stop();
}

TEST(FunctionScheduler, SchedulerRestart) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(10);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  fs.stop();

  ASSERT_EQ(counter, 2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(25));
  fs.stop();

  ASSERT_EQ(counter, 4);
}

TEST(FunctionScheduler, ConcurrentJobScheduling) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };

  c10::FunctionScheduler fs;
  fs.start();

  const int num_threads = 10;
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&fs, &function]() {
      for (int j = 0; j < 10; ++j) {
        fs.scheduleJob(function, std::chrono::milliseconds(10));
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  fs.stop();

  ASSERT_EQ(counter, num_threads * 10);
}

TEST(FunctionScheduler, ConcurrentJobRemoval) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };

  c10::FunctionScheduler fs;
  fs.start();

  const int num_jobs = 10;
  std::vector<int> job_ids;
  job_ids.reserve(num_jobs);
  for (int i = 0; i < num_jobs; ++i) {
    job_ids.push_back(fs.scheduleJob(function, std::chrono::milliseconds(10)));
  }

  std::vector<std::thread> threads;
  threads.reserve(num_jobs);
  for (int i = 0; i < num_jobs; ++i) {
    threads.emplace_back(
        [&fs, job_id = job_ids[i]]() { fs.removeJob(job_id); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(15));
  fs.stop();

  ASSERT_EQ(counter, 0);
}

TEST(FunctionScheduler, JobExceptions) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() {
    counter++;
    throw std::runtime_error("Test exception");
  };

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, std::chrono::milliseconds(1));
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  fs.stop();

  ASSERT_GE(counter, 1);
}

TEST(FunctionScheduler, RunImmediately) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(3);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_EQ(counter, 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(4));
  ASSERT_EQ(counter, 2);
  fs.stop();
}

TEST(FunctionScheduler, RunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(1);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  fs.stop();

  ASSERT_EQ(counter, 2);
}

TEST(FunctionScheduler, RunLimitReset) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(1);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 5);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // reset run limit
  fs.stop();
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  fs.stop();

  ASSERT_EQ(counter, 10);
}

TEST(FunctionScheduler, ImmediateJobWithRunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(1);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true, 3);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  fs.stop();

  ASSERT_EQ(counter, 3);
}

TEST(FunctionScheduler, PauseWhileRunning) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(2);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(7));
  fs.pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(6));
  fs.resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  fs.stop();

  ASSERT_EQ(counter, 4);
}
