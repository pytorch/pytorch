#include <c10/util/Exception.h>
#include <c10/util/FunctionScheduler.h>

#include <gtest/gtest.h>

TEST(Job, Initialization) {
  std::function<void()> function = []() {};
  std::chrono::microseconds interval(10);

  c10::Job j(function, interval);

  EXPECT_EQ(j.interval(), interval);
}

TEST(Job, Run) {
  bool ran = false;
  std::function<void()> function = [&ran]() { ran = true; };
  std::chrono::microseconds interval(10);

  c10::Job j(function, interval);
  j.run();

  EXPECT_TRUE(ran);
}

TEST(Run, Initialization) {
  int job_id = 1;
  auto time = std::chrono::steady_clock::now();

  c10::Run r(job_id, time);

  EXPECT_EQ(r.job_id(), job_id);
  EXPECT_EQ(r.time(), time);
}

TEST(Run, gt) {
  int job_id1 = 1;
  int job_id2 = 2;
  auto time1 = std::chrono::steady_clock::now();
  auto time2 = time1 + std::chrono::milliseconds(10);

  auto r1 = c10::Run(job_id1, time1);
  auto r2 = c10::Run(job_id2, time2);

  EXPECT_TRUE(c10::Run::gt(r2, r1));
  EXPECT_FALSE(c10::Run::gt(r1, r2));
}

TEST(FunctionScheduler, Initialization) {
  c10::FunctionScheduler fs;

  EXPECT_FALSE(fs.isRunning());
  EXPECT_EQ(fs.currentId(), 0);
}

TEST(FunctionScheduler, ScheduleJob) {
  std::function<void()> function = []() {};
  std::chrono::seconds interval(10);

  c10::FunctionScheduler fs;
  int job_id = fs.scheduleJob(function, interval);

  EXPECT_EQ(job_id, 0);
  EXPECT_EQ(fs.currentId(), 1);
}

TEST(FunctionScheduler, RemoveJob) {
  std::function<void()> function = []() {};
  std::chrono::seconds interval(10);

  c10::FunctionScheduler fs;

  int job_id = fs.scheduleJob(function, interval);
  EXPECT_EQ(job_id, 0);
  EXPECT_TRUE(fs.removeJob(job_id));
}

TEST(FunctionScheduler, RemoveNonExistentJob) {
  c10::FunctionScheduler fs;
  EXPECT_FALSE(fs.removeJob(0));
}

TEST(FunctionScheduler, RemoveFirstQueuedJob) {
  // This test verifies that the FunctionScheduler correctly handles the removal
  // of a scheduled job and ensures that the next job is executed in its place.
  // It specifically tests the FunctionScheduler::getNextWaitTime() method to
  // ensure that it properly skips over jobs that have been removed from the
  // queue.

  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function0 = []() {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  };
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::seconds interval0(1);
  std::chrono::milliseconds interval1(200);
  std::chrono::milliseconds interval2(400);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function0, interval0, true, 1);
  int job_id1 = fs.scheduleJob(function1, interval1);
  int job_id2 = fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  fs.removeJob(job_id1);
  std::this_thread::sleep_for(std::chrono::milliseconds(600));
  fs.stop();

  EXPECT_FALSE(yes1);
  EXPECT_TRUE(yes2);
}

TEST(FunctionScheduler, StartAndStop) {
  c10::FunctionScheduler fs;

  fs.start();
  ASSERT_TRUE(fs.isRunning());

  fs.stop();
  EXPECT_FALSE(fs.isRunning());
}

TEST(FunctionScheduler, RunJobWithZeroInterval) {
  bool ran = false;
  std::function<void()> function = [&ran]() { ran = true; };
  std::chrono::seconds interval(0);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fs.stop();

  EXPECT_TRUE(ran);
}

TEST(FunctionScheduler, RunJobWithInterval) {
  std::atomic<int> count = 0;
  std::function<void()> function = [&count]() { count++; };
  std::chrono::milliseconds interval(200);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  ASSERT_EQ(count, 0);

  std::this_thread::sleep_for(std::chrono::milliseconds(200)); // 250
  ASSERT_EQ(count, 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 300
  ASSERT_EQ(count, 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(150)); // 450
  EXPECT_EQ(count, 2);
  fs.stop();
}

TEST(FunctionScheduler, RemoveJobWhileRunning) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(400);

  c10::FunctionScheduler fs;
  int job_id = fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  fs.removeJob(job_id);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.stop();

  EXPECT_EQ(counter, 2);
}

TEST(FunctionScheduler, RunMultipleJobs) {
  std::atomic<int> counter = 0;
  std::function<void()> function1 = [&counter]() { counter++; };
  std::function<void()> function2 = [&counter]() { counter += 2; };
  std::chrono::milliseconds interval1(200);
  std::chrono::milliseconds interval2(400);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(550));
  fs.stop();

  EXPECT_EQ(counter, 4);
}

TEST(FunctionScheduler, RunMultipleJobsWithSameInterval) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::milliseconds interval1(200);
  std::chrono::milliseconds interval2(200);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(350));
  fs.stop();

  EXPECT_EQ(counter1, 1);
}

TEST(FunctionScheduler, ScheduleJobAfterCurrentlyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::milliseconds interval1(500);
  std::chrono::milliseconds interval2(400);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_TRUE(yes1);
  EXPECT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  EXPECT_TRUE(yes1);
  EXPECT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeCurrenltyWaiting) {
  std::atomic<bool> yes1 = false;
  std::atomic<bool> yes2 = false;
  std::function<void()> function1 = [&yes1]() { yes1 = true; };
  std::function<void()> function2 = [&yes2]() { yes2 = true; };
  std::chrono::milliseconds interval1(1000);
  std::chrono::milliseconds interval2(200);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.scheduleJob(function2, interval2);
  EXPECT_FALSE(yes1);
  EXPECT_FALSE(yes2);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  EXPECT_FALSE(yes1);
  EXPECT_TRUE(yes2);
  fs.stop();
}

TEST(FunctionScheduler, ScheduleJobBeforeAndAfterCurrenltyWaiting) {
  std::atomic<int> counter1 = 0;
  std::atomic<int> counter2 = 0;
  std::function<void()> function1 = [&counter1]() { counter1++; };
  std::function<void()> function2 = [&counter2]() { counter2++; };
  std::chrono::milliseconds interval1(600);
  std::chrono::milliseconds interval2(200);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.scheduleJob(function2, interval2);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  EXPECT_EQ(counter1, 1);
  EXPECT_EQ(counter2, 2);
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

  EXPECT_EQ(counter, 4);
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
        fs.scheduleJob(function, std::chrono::milliseconds(10), false, 5);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  fs.stop();

  EXPECT_EQ(counter, num_threads * 10 * 5);
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
    job_ids.push_back(fs.scheduleJob(function, std::chrono::milliseconds(100)));
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

  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fs.stop();

  EXPECT_EQ(counter, 0);
}

TEST(FunctionScheduler, JobExceptions) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() {
    counter++;
    throw std::runtime_error("Test exception");
  };

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, std::chrono::milliseconds(1), true, 1);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  fs.stop();

  EXPECT_GE(counter, 1);
}

TEST(FunctionScheduler, RunImmediately) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(300);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(counter, 1);

  std::this_thread::sleep_for(std::chrono::milliseconds(450));
  ASSERT_EQ(counter, 2);
  fs.stop();
}

TEST(FunctionScheduler, RunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 2);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  fs.stop();

  EXPECT_EQ(counter, 2);
}

TEST(FunctionScheduler, RunLimitReset) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, false, 3);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(700));

  // reset run limit
  fs.stop();
  fs.start();

  std::this_thread::sleep_for(std::chrono::milliseconds(700));
  fs.stop();

  EXPECT_EQ(counter, 6);
}

TEST(FunctionScheduler, ImmediateJobWithRunLimit) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval, true, 3);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  fs.stop();

  EXPECT_EQ(counter, 3);
}

TEST(FunctionScheduler, PauseWhileRunning) {
  std::atomic<int> counter = 0;
  std::function<void()> function = [&counter]() { counter++; };
  std::chrono::milliseconds interval(100);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);

  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
  fs.pause();
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  fs.resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  fs.stop();

  EXPECT_EQ(counter, 4);
}

TEST(FunctionScheduler, InvalidJobInterval) {
  std::function<void()> function = []() { return; };
  std::chrono::milliseconds interval(-1);
  c10::FunctionScheduler fs;

  EXPECT_THROW(
      {
        try {
          fs.scheduleJob(function, interval);
        } catch (const c10::Error& e) {
          ASSERT_EQ("Job interval must be positive.", e.msg());
          throw;
        }
      },
      c10::Error);
}

TEST(FunctionScheduler, InvalidFunction) {
  std::function<void()> function;
  std::chrono::milliseconds interval(1);
  c10::FunctionScheduler fs;

  EXPECT_THROW(
      {
        try {
          fs.scheduleJob(function, interval);
        } catch (const c10::Error& e) {
          ASSERT_EQ("Job function can't be null.", e.msg());
          throw;
        }
      },
      c10::Error);
}

TEST(FunctionScheduler, InvalidRunLimit) {
  std::function<void()> function = []() { return; };
  std::chrono::milliseconds interval(1);
  c10::FunctionScheduler fs;

  EXPECT_THROW(
      {
        try {
          fs.scheduleJob(function, interval, false, 0);
        } catch (const c10::Error& e) {
          ASSERT_EQ(
              "Job run limit must be greater than 0 or FunctionScheduler::RUN_FOREVER (-1).",
              e.msg());
          throw;
        }
      },
      c10::Error);
}
