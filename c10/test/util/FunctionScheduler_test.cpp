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

TEST(Run, lt) {
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
  std::chrono::seconds interval(3);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_FALSE(ran);

  std::this_thread::sleep_for(std::chrono::seconds(5));
  fs.stop();

  ASSERT_TRUE(ran);
}

TEST(FunctionScheduler, RemoveJobWhileRunning) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(200);

  c10::FunctionScheduler fs;
  int job_id = fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  fs.removeJob(job_id);
  std::this_thread::sleep_for(std::chrono::seconds(1));
  fs.stop();

  ASSERT_EQ(counter, 5);
}

TEST(FunctionScheduler, RunMultipleJobs) {
  std::atomic<int> counter = 0;
  std::function<void()> function1 = [&counter]() { counter++; };
  std::function<void()> function2 = [&counter]() { counter += 2; };
  std::chrono::seconds interval1(2); // 2s, 4s
  std::chrono::seconds interval2(4); // 4s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(5));
  fs.stop();

  ASSERT_EQ(counter, 4);
}

TEST(FunctionScheduler, RunMultipleJobsWithSameInterval) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::seconds interval1(2); // 2s
  std::chrono::seconds interval2(2); // 2s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(3)); // 3s
  fs.stop();

  ASSERT_EQ(counter1, 1);
  ASSERT_EQ(counter1, 1);
}

TEST(FunctionScheduler, ScheduleJobAfterCurrentlyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::seconds interval1(5); // 5s
  std::chrono::seconds interval2(4); // 3+4s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::seconds(3)); // 3s
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::seconds(3)); // 6s
  ASSERT_TRUE(yes1);
  ASSERT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::seconds(2)); // 8s
  ASSERT_TRUE(yes1);
  ASSERT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeCurrenltyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::seconds interval1(10); // 10s
  std::chrono::seconds interval2(2); // 3+2s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::seconds(3)); // 3s
  fs.scheduleJob(function2, interval2);
  ASSERT_FALSE(yes1);
  ASSERT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::seconds(3)); // 6s
  ASSERT_FALSE(yes1);
  ASSERT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeAndAfterCurrenltyWaiting) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::seconds interval1(6); // 6s
  std::chrono::seconds interval2(2); // 3+2s, 3+4s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::seconds(3)); // 3s
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::seconds(5)); // 8s
  ASSERT_EQ(counter1, 1);
  ASSERT_EQ(counter2, 2);
  fs.stop();
}

TEST(FunctionScheduler, SchedulerRestart) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  fs.stop();

  ASSERT_EQ(counter, 2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  fs.stop();

  ASSERT_EQ(counter, 4);
}

TEST(FunctionScheduler, ConcurrentJobScheduling) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };

  c10::FunctionScheduler fs;
  fs.start();

  const int num_threads = 10;
  std::vector<std::thread> threads(num_threads);
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

  std::this_thread::sleep_for(std::chrono::seconds(1));
  fs.stop();

  ASSERT_GE(counter.load(), num_threads * 10);
}

TEST(FunctionScheduler, ConcurrentJobRemoval) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };

  c10::FunctionScheduler fs;
  fs.start();

  const int num_jobs = 10;
  std::vector<int> job_ids(num_jobs);
  for (int i = 0; i < num_jobs; ++i) {
    job_ids.push_back(fs.scheduleJob(function, std::chrono::seconds(1)));
  }

  std::vector<std::thread> threads(num_jobs);
  for (int i = 0; i < num_jobs; ++i) {
    threads.emplace_back(
        [&fs, job_id = job_ids[i]]() { fs.removeJob(job_id); });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));
  fs.stop();

  ASSERT_EQ(counter.load(), 0);
}

TEST(FunctionScheduler, JobExceptions) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() {
    counter++;
    throw std::runtime_error("Test exception");
  };

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, std::chrono::milliseconds(100));
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.stop();

  ASSERT_GE(counter.load(), 1);
}

TEST(FuntionScheduler, RunImmediately) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::seconds interval(3);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  ASSERT_EQ(counter, 1);

  std::this_thread::sleep_for(std::chrono::seconds(3));
  ASSERT_EQ(counter, 2);
  fs.stop();
}

TEST(FunctionScheduler, RunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(50);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 5);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  fs.stop();

  ASSERT_EQ(counter, 5);
}

TEST(FunctionScheduler, RunLimitReset) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(50);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 5);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // reset run limit
  fs.stop();
  fs.start();

  std::this_thread::sleep_for(std::chrono::seconds(1));
  fs.stop();

  ASSERT_EQ(counter, 10);
}

TEST(FunctionScheduler, ImmediateJobWithRunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true, 3);

  fs.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  fs.stop();

  ASSERT_EQ(counter, 3);
}
