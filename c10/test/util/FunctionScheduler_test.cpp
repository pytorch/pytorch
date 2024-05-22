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

  ASSERT_TRUE(c10::Run::lt(r1, r2));
  ASSERT_FALSE(c10::Run::lt(r2, r1));
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

TEST(FunctionScheduler, StartAndStop) {
  c10::FunctionScheduler fs;

  fs.start();
  ASSERT_TRUE(fs.isRunning());

  fs.stop();
  ASSERT_FALSE(fs.isRunning());
}

TEST(FunctionScheduler, RunJobImmediately) {
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
  std::chrono::seconds interval(15);

  c10::FunctionScheduler fs;
  fs.scheduleJob(function, interval);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_FALSE(ran);

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  fs.stop();

  ASSERT_TRUE(ran);
}

TEST(FunctionScheduler, RunMultipleJobs) {
  int counter = 0; // TODO should be atomic if multiple threads exist?
  std::function<void()> function1 = [&counter]() { counter++; };
  std::function<void()> function2 = [&counter]() { counter += 2; };
  std::chrono::seconds interval1(10); // 10s, 20s
  std::chrono::seconds interval2(15); // 15s

  c10::FunctionScheduler fs;
  fs.scheduleJob(function1, interval1);
  fs.scheduleJob(function2, interval2);
  fs.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(22));
  fs.stop();

  ASSERT_EQ(counter, 4);
}
