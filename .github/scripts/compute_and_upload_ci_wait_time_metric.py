import os
import time
import rockset
import pandas as pd
import boto3 # type: ignore[import]
from botocore.exceptions import ClientError  # type: ignore[import]
from decimal import Decimal


TABLE_NAME = "torchci-metrics-ci-wait-time"
dynamodb = boto3.resource('dynamodb', region_name="us-east-1")

# Make the results more consistent, and ensure proper typing of the columns
def normalize_workflow_runs(records):
    records['start_time'] = pd.to_datetime(records['start_time'])
    records['end_time'] = pd.to_datetime(records['end_time'])

    records['pr_number'] = records['pr_number'].astype(int)
    records['workflow_run_id'] = records['workflow_run_id'].astype(int)
    return records

def query_workflows_from_rockset() -> pd.DataFrame:
    rs = rockset.RocksetClient(
        host="api.usw2a1.rockset.com", api_key=os.environ["ROCKSET_API_KEY"]
    )
    params = [
        rockset.models.QueryParameter(
            name="from_days_ago",
            type="int",
            value="15",
        ),
        rockset.models.QueryParameter(
            name="to_days_ago",
            type="int",
            value="0",
        ),
    ]

    data = []
    page_num = 1
    start_time = time.time()
    for page in rockset.QueryPaginator(
        rs,
        rs.QueryLambdas.execute_query_lambda(
            query_lambda="completed_pr_jobs_aggregate",
            version="01fc2a697ae85856",
            workspace="metrics",
            parameters=params,
            paginate=True,
            initial_paginate_response_doc_count=10000,
        )
    ):
        print(f"Got page {page_num}")
        page_num += 1
        data += page
    
    print(f"Total rows received: {len(data)}")
    end_time = time.time()
    print(f"Total query time: {end_time - start_time} seconds ")

    results_df = pd.json_normalize(data)

    return normalize_workflow_runs(results_df)


def remove_cancelled_jobs(df):
    # Cancelled jobs can fall into one of three categories:
    # 1. The job timed out
    # 2. The run was cancelled by a user's push before any job failed
    # 3. The run was cancelled by a user's push after a job failed

    # Identify the cancellations that were timeouts. It's a timeout if the job was cancelled after 5hrs, 
    df['was_timeout'] = df.apply(lambda row: row['was_cancelled'] and row['duration_mins'] >= 300, axis=1)

    # Consider timeouts to be a type of failures. 
    df['conclusion'] = df.apply(lambda row: 'failure' if row['was_timeout'] else row['conclusion'], axis=1)

    # For non-timeout cancellations, if the workflows were cancelled _after_ a job already failed, then
    # we say the failure signal gave the "end time" since it provided an actionable signal.
    # This means we can ignore the 'cancelled' jobs for such workflow runs.
    # Thus, if the job failed, and later it's workflow was cancelled then we ignore those cancelled jobs
    df = df[~( (df['conclusion'] == 'cancelled') & (df['sha'].isin(df[df['conclusion'] == 'failure']['sha'])))]

    # Now the remaining cancelled jobs are the ones cancelled due to additional commits being pushed
    # _before_ a job failed. We can ignore these entire commits since they did't provide any actionable signal.
    # Thus we drop all shas that still contain a cancelled job
    df = df[~df['sha'].isin(df[df['conclusion'] == 'cancelled']['sha'])]
    
    return df


# For each workflow run, set the start_time for the group to the earliest start_time in that group
# since the first job shows when the user started waiting on CI
def normalize_start_times(df):
    # We track each run attempt in a workflow run separately. 
    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    df = df.merge(df_grouped['start_time'].min(), on=['sha', 'run_attempt', 'workflow_run_id'], how='left', suffixes=('_orig', ''))
    
    # Update the duration accordingly
    df['duration_mins'] = round((df['end_time'] - df['start_time']).dt.total_seconds() / 60)

    return df

def ignore_failures_from_retried_jobs(df):
    # If this isn't the last run attempt, then the dev was blocked (unable to retry) until the last job completed.
    # Simulate that experience by having all conclusions for jobs that were retried to 'success', since 
    # we don't want to stop the clock early
    df['conclusion'] = df.apply(lambda row: 'success' if row['run_attempt'] != row['total_attempts'] else row['conclusion'], axis=1)
    return df


# Within a run, if there are both successful and failed jobs, the successful jobs are irrelevant. Let's remove them
def remove_irrelevant_success_jobs(df):
    df = normalize_start_times(df) # Ensure we keep the earliest start times
    print ("normalized start times")

    # If a sha has a failuring job, then remove 'success' jobs for that sha
    # since the failuring row will set the end time
    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    for (sha, run_attempt, workflow_run_id), group in df_grouped:
        if group[group['conclusion'] == 'failure'].shape[0] > 0:
            df = df[~((df['sha'] == sha) & (df['run_attempt'] == run_attempt) & (df['workflow_run_id'] == workflow_run_id) & (df['conclusion'] == 'success'))]

    # If a run attempt has multiple 'success' jobs, preserve the one that ended last
    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    for (sha, run_attempt, workflow_run_id), group in df_grouped:
        if group[group['conclusion'] == 'successs'].shape[0] > 1:
            df = df[~((df['sha'] == sha) & (df['run_attempt'] == run_attempt) & (df['workflow_run_id'] == workflow_run_id) & (df['conclusion'] == 'success') & (df['end_time'] != group['end_time'].max()))]

    return df

# Within a run, if there are multiple batches of failed jobs, the one that ends first gave the first signal
def remove_irrelevant_failure_jobs(df):
    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    for (sha, run_attempt, workflow_run_id), group in df_grouped:
        if group[group['conclusion'] == 'failure'].shape[0] > 1:
            df = df[~((df['sha'] == sha) & (df['run_attempt'] == run_attempt) & (df['workflow_run_id'] == workflow_run_id) & (df['conclusion'] == 'failure') & (df['end_time'] != group['end_time'].min()))]

    return df


# Some jobs are extra funky. Remove them from our data to donoise it and decomplicate the logic.
def discard_weird_cases(df):
    # Multiple PRs could potentially share the same commit in their history
    #  (this only affected 4 PRs over a 6 month window)
    # Discard those PRs.
    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    for _, group in df_grouped:
        if group.shape[0] > 1:
            df = df[~df['pr_number'].isin(group['pr_number'])]
    
    # The same workflow can get accidentally triggered against the same commit more than 
    #  once (I'm not talking about retries here).
    # We remove duplicate workflows that were run against the same commit, keeping only 
    #  the workflow triggered first. 
    # They'll have different workflow_run_ids but the same workflow name
    df_grouped = df.groupby(['sha', 'workflow_name', 'run_attempt'])
    for (sha, workflow_name, run_attempt), group in df_grouped:
        if group.shape[0] > 1:
            df = df[~((df['sha'] == sha) & (df['workflow_name'] == workflow_name) & (df['run_attempt'] == run_attempt) & (df['start_time'] != group['start_time'].min()))]

    return df


class OverlapableTimeSpan:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

    def overlaps_with(self, timespan):
        return self.start_time < timespan.end_time and self.end_time > timespan.start_time

    def union_with(self, timespan):
        if self.overlaps_with(timespan):
            self.start_time = min(self.start_time, timespan.start_time)
            self.end_time = max(self.end_time, timespan.end_time)
            return True
        else:
            return False
        
    def get_duration_mins(self):
        return round((self.end_time - self.start_time).total_seconds() / 60)


# Combine the time spans across all jobs in a PR to get the PR's total duration.
# Some workflows run in parallel, e.g. pull/trunk, and we don't want to double count time 
# spent waiting on them
def get_duration_for_pr(pr_df_group):
    logged_timespans = []
    pr_df_group = pr_df_group.sort_values(by=['start_time'])
    
    # Find the overlapping timespans and combine them
    for _, row in pr_df_group.iterrows():
        timespan = OverlapableTimeSpan(row['start_time'], row['end_time'])
        found_match = False
        for existing_span in logged_timespans:
            if existing_span.overlaps_with(timespan):
                # Found an overlapping set of timestamps
                existing_span.union_with(timespan)
                found_match = True
                break

        if not found_match:
            logged_timespans.append(timespan)

    duration = 0
    for timespan in logged_timespans:
        duration += timespan.get_duration_mins()

    return duration

def validate_all_data_has_been_deduped(df):
    unduped_jobs = pd.DataFrame(columns=['sha', 'run_attempt', 'workflow_run_id'])

    df_grouped = df.groupby(['sha', 'run_attempt', 'workflow_run_id'])
    for (sha, run_attempt, workflow_run_id), group in df_grouped:
        if group.shape[0] > 1:
            print ("Found duplicate rows for sha: %s, run_attempt: %s, workflow_run_id: %s" % (sha, run_attempt, workflow_run_id))
            print (group)
            unduped_jobs = unduped_jobs.append(group)

    if unduped_jobs.shape[0] > 0:
        raise Exception("Found unduped jobs for records: %s" % unduped_jobs.to_string())
    else:
        print ("All data has been deduped")

# Now that jobs with the same sha, run_attempt, and workflow_run_id have all 
#  been deduped, we only have a single conclusion for each workflow run attempt
# We're interested in:
#  1. The wall clock time spent waiting on the CI per PR
#  2. The number of commits it had
def get_pr_level_stats(df):
    # Now, all run attempts should have exactly one row. If not, we missed a corner case
    validate_all_data_has_been_deduped(df)

    df_results = pd.DataFrame(columns=['pr_number', 'duration_mins', 'end_time'])
    
    df_grouped = df.groupby('pr_number')    
    for (pr_number), group in df_grouped:        
        pr_ci_duration = get_duration_for_pr(group)
        num_commits = group['sha'].nunique()
        pr_start_time = group['start_time'].min()
        pr_end_time = group['end_time'].max()
        week = pr_end_time.to_period('W').start_time

        # Stats for this PR
        df_results = pd.concat([df_results, pd.DataFrame({
            'pr_number': [pr_number], 
            'duration_mins': [pr_ci_duration],
            'start_time': [pr_start_time],
            'end_time': [pr_end_time],
            'num_commits': [num_commits],
            'week': [week],
        })])
    
    return df_results

def get_pr_stats() -> pd.DataFrame:
    runs = query_workflows_from_rockset()

    # Now we have all the raw data and it's ime to start filtering it
    #
    # We need to figure out the effective start/end times for each workflow run 
    # even though that workflows could have concluded in different ways
    #
    # Definitions:
    #   Start time: When the CI jobs were requested (usually when the dev pushes the commit, can
    #                also be when a label is applied to a commit, or when a workflow is rerun)
    #   End time:   When the dev has an actionable signal. Actionable meaning it's either an
    #                all green signal or it's the first red signal 
    #
    # We filter out jobs that don't contribute to determining the end time of the workflow run
    #
    # The rules used to figure that out:
    #   1. If jobs in a workflow are cancelled before any jobs in CI failed, then the dev
    #       must have pushed a new commit before getting a signal from CI. 
    #       Since they were not blocked on CI, we ignore that entire commit's workflow.
    #       The only exception is when the job duration was >5hrs, in which case they actually
    #       ran into a timeout
    #   2. If a workflow is retried after it failed, then count the entire duration of the workflow
    #       since the dev had to wait for it to complete before doing the retry
    #   2. If any job fails on the final retry attempt then the dev has gotten a signal. Jobs that
    #       succeed or fail after that one are irrelevant
    #   3. If all jobs complete successfully, the last job that completed is what gives the
    #       signal (i.e. sets the end time)
    runs = remove_cancelled_jobs(runs)
    runs = normalize_start_times(runs) # Ensure all jobs with a workflow have the same start time
    runs = ignore_failures_from_retried_jobs(runs)
    runs = remove_irrelevant_success_jobs(runs)
    runs = remove_irrelevant_failure_jobs(runs)
    runs = discard_weird_cases(runs)

    return get_pr_level_stats(runs)

def table_exists(table_name: str) -> bool:
    """
    Determines whether a table exists. As a side effect, stores the table in
    a member variable.
    """
    exists = None
    try:
        table = dynamodb.Table(table_name)
        table.load()
        exists = True
        print(f"Table '{table_name}' exists")
    except ClientError as err:
        if err.response["Error"]["Code"] == "ResourceNotFoundException":
            print(f"Table '{table_name}' does not exist")
            exists = False
        else:
            print("Unknown error")
            print(err.response)
    return exists

def upload_stats(pr_stats):
    print(pr_stats.head(30))

    print(f"Uploading data to {TABLE_NAME}")

    statsTable = dynamodb.Table(TABLE_NAME)

    for index, row in pr_stats.iterrows():
        statsTable.update_item(
            Key={
                'dynamoKey': f"{row['week'].isoformat()}_{row['pr_number']}",
            },
            UpdateExpression="set pr_number = :p duration_mins = :d, start_time = :s, end_time = :e, num_commits = :n, week = :w",
            ExpressionAttributeValues={
                ':p': row['pr_number'],
                ':d': int(row['duration_mins']),
                ':s': row['start_time'].isoformat(),
                ':e': row['end_time'].isoformat(),
                ':n': int(row['num_commits']),
                ':w': row['week'].isoformat(),
            },
            ReturnValues="UPDATED_NEW"
        )

    print(f"Finished uploading data to {TABLE_NAME}")


def main() -> None:
    pr_stats = get_pr_stats()
    upload_stats(pr_stats)

if __name__ == "__main__":
    main()