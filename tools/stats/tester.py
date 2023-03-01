from upload_stats_lib import upload_to_s3
if __name__ == "__main__":
    upload_to_s3(1, 1, "testing123", [{"a": 1},{"a": 2}])