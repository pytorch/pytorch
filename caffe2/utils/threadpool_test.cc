#include <gtest/gtest.h>

#include "caffe2/utils/threadpool/pthreadpool.h"
#include "caffe2/utils/threadpool/pthreadpool_impl.h"
#include "caffe2/utils/threadpool/ThreadPool.h"


using caffe2::ThreadPool;


const size_t itemsCount1D = 1024;

TEST(ThreadPool, CreateAndDestroy) {
	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
}

static void computeNothing1D(void*, size_t) {
}

TEST(ThreadPool, Basic) {
	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
	pthreadpool pThreadPool(threadPool.get());
	pthreadpool_compute_1d(&pThreadPool, computeNothing1D, NULL, itemsCount1D);
}

static void checkRange1D(void*, size_t itemId) {
	EXPECT_LT(itemId, itemsCount1D);
}

TEST(ThreadPool, ValidRange) {
	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
	pthreadpool pThreadPool(threadPool.get());
	pthreadpool_compute_1d(&pThreadPool, checkRange1D, NULL, itemsCount1D);
}

static void setTrue1D(bool indicators[], size_t itemId) {
	indicators[itemId] = true;
}

TEST(ThreadPool, AllItemsProcessed) {
	bool processed[itemsCount1D];
	memset(processed, 0, sizeof(processed));

	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
	pthreadpool pThreadPool(threadPool.get());
	pthreadpool_compute_1d(&pThreadPool, reinterpret_cast<pthreadpool_function_1d_t>(setTrue1D), processed, itemsCount1D);
	for (size_t itemId = 0; itemId < itemsCount1D; itemId++) {
		EXPECT_TRUE(processed[itemId]) << "Item " << itemId << " not processed";
	}
}

static void increment1D(int counters[], size_t itemId) {
	counters[itemId] += 1;
}

TEST(ThreadPool, EachItemProcessedOnce) {
	int processedCount[itemsCount1D];
	memset(processedCount, 0, sizeof(processedCount));

	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
	pthreadpool pThreadPool(threadPool.get());
	pthreadpool_compute_1d(&pThreadPool, reinterpret_cast<pthreadpool_function_1d_t>(increment1D), processedCount, itemsCount1D);
	for (size_t itemId = 0; itemId < itemsCount1D; itemId++) {
		EXPECT_EQ(1, processedCount[itemId]) << "Item " << itemId << " processed " << processedCount[itemId] << " times";
	}
}

TEST(ThreadPool, EachItemProcessedMultipleTimes) {
	int processedCount[itemsCount1D];
	memset(processedCount, 0, sizeof(processedCount));
	const size_t iterations = 100;

	std::unique_ptr<ThreadPool> threadPool = ThreadPool::defaultThreadPool();
	pthreadpool pThreadPool(threadPool.get());
	for (size_t iteration = 0; iteration < iterations; iteration++) {
		pthreadpool_compute_1d(&pThreadPool, reinterpret_cast<pthreadpool_function_1d_t>(increment1D), processedCount, itemsCount1D);
	}
	for (size_t itemId = 0; itemId < itemsCount1D; itemId++) {
		EXPECT_EQ(iterations, processedCount[itemId]) << "Item " << itemId << " processed " << processedCount[itemId] << " times";
	}
}
