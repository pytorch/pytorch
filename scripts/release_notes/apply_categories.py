# Quick scipt to apply categorized items to the
# base commitlist . Useful if you are refactoring any code
# but want to keep the previous data on categories

import commitlist
import csv

category_csv = "results/category_data.csv"
commitlist_csv = "results/commitlist.csv"

with open(category_csv, "r") as category_data:
    reader = csv.DictReader(category_data, commitlist.commit_fields)
    rows = list(reader)
    category_map = {row["commit_hash"]: row["category"] for row in rows}

with open(commitlist_csv, "r") as commitlist_data:
    reader = csv.DictReader(commitlist_data, commitlist.commit_fields)
    commitlist_rows = list(reader)

for row in commitlist_rows:
    hash = row["commit_hash"]
    if hash in category_map and category_map[hash] != "Uncategorized":
        row["category"] = category_map[hash]

with open(commitlist_csv, "w") as commitlist_write:
    writer = csv.DictWriter(commitlist_write, commitlist.commit_fields)
    writer.writeheader()
    writer.writerows(commitlist_rows)
