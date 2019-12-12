#!/usr/bin/env python

import argparse
import datetime
import boto3
import pytz
import sys


def repos(client):
    paginator = client.get_paginator("describe_repositories")
    pages = paginator.paginate(registryId="308535385114")
    for page in pages:
        for repo in page["repositories"]:
            yield repo


def images(client, repository):
    paginator = client.get_paginator("describe_images")
    pages = paginator.paginate(
        registryId="308535385114", repositoryName=repository["repositoryName"]
    )
    for page in pages:
        for image in page["imageDetails"]:
            yield image


parser = argparse.ArgumentParser(description="Delete old Docker tags from registry")
parser.add_argument(
    "--dry-run", action="store_true", help="Dry run; print tags that would be deleted"
)
parser.add_argument(
    "--keep-stable-days",
    type=int,
    default=14,
    help="Days of stable Docker tags to keep (non per-build images)",
)
parser.add_argument(
    "--keep-unstable-days",
    type=int,
    default=1,
    help="Days of unstable Docker tags to keep (per-build images)",
)
parser.add_argument(
    "--filter-prefix",
    type=str,
    default="",
    help="Only run cleanup for repositories with this prefix",
)
parser.add_argument(
    "--ignore-tags",
    type=str,
    default="",
    help="Never cleanup these tags (comma separated)",
)
args = parser.parse_args()

if not args.ignore_tags or not args.filter_prefix:
    print(
        """
Missing required arguments --ignore-tags and --filter-prefix

You must specify --ignore-tags and --filter-prefix to avoid accidentally
pruning a stable Docker tag which is being actively used.  This will
make you VERY SAD.  So pay attention.

First, which filter-prefix do you want?  The list of valid prefixes
is in jobs/private.groovy under the 'docker-registry-cleanup' job.
You probably want either pytorch or caffe2.

Second, which ignore-tags do you want?  It should be whatever the most
up-to-date DockerVersion for the repository in question is.  Follow
the imports of jobs/pytorch.groovy to find them.
"""
    )
    sys.exit(1)

client = boto3.client("ecr", region_name="us-east-1")
stable_window = datetime.timedelta(days=args.keep_stable_days)
unstable_window = datetime.timedelta(days=args.keep_unstable_days)
now = datetime.datetime.now(pytz.UTC)
ignore_tags = args.ignore_tags.split(",")


def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(chunkable), n):
        yield chunkable[i : i + n]


for repo in repos(client):
    repositoryName = repo["repositoryName"]
    if not repositoryName.startswith(args.filter_prefix):
        continue

    # Keep list of image digests to delete for this repository
    digest_to_delete = []

    print(repositoryName)
    for image in images(client, repo):
        tags = image.get("imageTags")
        if not isinstance(tags, (list,)) or len(tags) == 0:
            continue

        tag = tags[0]
        # new images build on circle ci use workflow ID as tag, which has 4 "-"
        if tag.isdigit() or tag.count("-") == 4:
            window = stable_window
        else:
            window = unstable_window

        created = image["imagePushedAt"].replace(tzinfo=pytz.UTC)
        age = now - created
        if tag in ignore_tags:
            print("Ignoring tag {} (age: {})".format(tag, age))
            continue
        if age < window:
            print("Not deleting manifest for tag {} (age: {})".format(tag, age))
            continue

        if args.dry_run:
            print("(dry run) Deleting manifest for tag {} (age: {})".format(tag, age))
        else:
            print("Deleting manifest for tag {} (age: {})".format(tag, age))
            digest_to_delete.append(image["imageDigest"])

    # Issue batch delete for all images to delete for this repository
    # Note that as of 2018-07-25, the maximum number of images you can
    # delete in a single batch is 100, so chunk our list into batches of
    # 100
    for c in chunks(digest_to_delete, 100):
        client.batch_delete_image(
            registryId="308535385114",
            repositoryName=repositoryName,
            imageIds=[{"imageDigest": digest} for digest in c],
        )
