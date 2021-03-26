from typing import Dict, Tuple, List

def calculate_shards(num_shards: int, tests: List[str], job_times: Dict[str, float]) -> List[Tuple[float, List[str]]]:
    filtered_job_times: Dict[str, float] = dict()
    unknown_jobs : List[str] = []
    for test in tests:
        if test in job_times:
            filtered_job_times[test] = job_times[test]
        else:
            unknown_jobs.append(test)

    # The following attempts to implement a partition approximation greedy algorithm
    # See more at https://en.wikipedia.org/wiki/Greedy_number_partitioning
    sorted_jobs = sorted(filtered_job_times, key=lambda j: filtered_job_times[j], reverse=True)
    sharded_jobs: List[Tuple[float, List[str]]] = [(0.0, []) for _ in range(num_shards)]
    for job in sorted_jobs:
        min_shard_index = sorted(range(num_shards), key=lambda i: sharded_jobs[i][0])[0]
        curr_shard_time, curr_shard_jobs = sharded_jobs[min_shard_index]
        curr_shard_jobs.append(job)
        sharded_jobs[min_shard_index] = (curr_shard_time + filtered_job_times[job], curr_shard_jobs)

    # Round robin the unknown jobs starting with the smallest shard
    index = sorted(range(num_shards), key=lambda i: sharded_jobs[i][0])[0]
    for job in unknown_jobs:
        sharded_jobs[index][1].append(job)
        index = (index + 1) % num_shards
    return sharded_jobs
