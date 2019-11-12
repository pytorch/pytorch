#! python3
import json
import aiohttp
import asyncio
import datetime
import os
from collections import defaultdict, OrderedDict
from statistics import mean


CIRCLE_TOKEN = os.environ.get("CIRCLE_TOKEN")
auth = aiohttp.BasicAuth(CIRCLE_TOKEN, "")


class AsyncJobGetter:
    """
    Basic CircleCI schema:
        - Pipeline: All CircleCI associated with a specific revision. Contains multiple workflows.
        - Workflow: A grouping of jobs on a specific revision.
            (examples: "build", "binarybuilds", "binarysmoketest")
        - Job: An individual unit of work
            (examples: pytorch_python_doc_push, pytorch_linux_xenial_py3_6_gcc7_build)
    """
    def __init__(self, session):
        self.session = session

    async def get_jobs(self, pages):
        """
        pages: number of pages of pipelines to retrieve. Each page has ~20 piplines.

        Returns:
            List[Dict[job_name => job json]]
        Each element of the list represents a pipeline. The dictionary is the
        jobs in the "build" workflow.
        """
        pipelines = []
        for i in range(pages):
            if i == 0:
                params = {"branch": "master"}
            else:
                # for
                params = {"page-token": next_page_token}

            page = await self._fetch(
                "https://circleci.com/api/v2/project/gh/pytorch/pytorch/pipeline",
                params=params,
            )
            next_page_token = page["next_page_token"]

            pipelines.extend(page["items"])
        return await self._get_pipelines(pipelines)

    async def _get_pipelines(self, pipelines):
        """
        Fan out across all pipelines
        """
        pipeline_ids = [pipeline["id"] for pipeline in pipelines]
        tasks = []
        for pipeline_id in pipeline_ids:
            tasks.append(self._fetch_jobs_from_pipeline(pipeline_id))
        return await asyncio.gather(*tasks)

    async def _fetch_jobs_from_pipeline(self, pipeline_id):
        """
        Fan out across all jobs in each pipeline.
        NOTE: Today we only have one workflow per pipeline, so we assume we
        can skip fanning out from pipeline -> workflows.
        """
        # TODO this will get deprecated soon in favor of pipeline/pipeline_id/workflow
        pipeline = await self._fetch(
            f"https://circleci.com/api/v2/pipeline/{pipeline_id}"
        )

        # TODO: This will not be correct if we start using more than workflow.
        # In that case, we'll need to search for "build"
        workflow_id = pipeline["workflows"][0]["id"]
        jobs = await self._fetch(
            f"https://circleci.com/api/v2/workflow/{workflow_id}/job"
        )
        assert jobs["next_page_token"] is None
        jobs = jobs["items"]
        tasks = []
        for job in jobs:
            if "job_number" not in job:
                # this job hasn't been started yet
                continue
            job_number = job["job_number"]
            job_task = self._fetch(
                f"https://circleci.com/api/v2/project/gh/pytorch/pytorch/job/{job_number}"
            )
            tasks.append(job_task)

        jobs = await asyncio.gather(*tasks)
        return {job["name"]: job for job in jobs}

    async def _fetch(self, url, params={}):
        async with self.session.get(
            url, headers={"Accept": "application/json"}, params=params
        ) as res:
            res = await res.text()
            return json.loads(res)


async def do_main():
    async with aiohttp.ClientSession(auth=auth) as session:
        job_getter = AsyncJobGetter(session)
        return await job_getter.get_jobs(pages=5)


if __name__ == "__main__":
    pipelines = asyncio.run(do_main())
    aggregated_times = defaultdict(list)
    for pipeline in pipelines:
        for job_name, job in pipeline.items():
            if job["duration"] is None:
                # happens if the job is still running
                continue
            aggregated_times[job_name].append(job["duration"])

    print(f"Looked at {len(aggregated_times)} total master commits")
    averaged_times = {name: mean(times) for name, times in aggregated_times.items()}

    averaged_times = OrderedDict(
        sorted(averaged_times.items(), key=lambda kv: kv[1], reverse=True)
    )
    for name, avg_time in averaged_times.items():
        avg_time = datetime.timedelta(milliseconds=avg_time)
        print(name.ljust(80), avg_time)
