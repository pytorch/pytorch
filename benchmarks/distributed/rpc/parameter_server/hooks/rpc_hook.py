from utils import process_bucket_with_remote_server


def rpc_hook(state, bucket):
    r"""
    A ddp communication hook that averages sparse and dense tensors using
    process_bucket_with_remote_server method.
    Args:
        state (object): maintains state during the training process
        bucket (GradBucket): gradient bucket
    """
    return process_bucket_with_remote_server(state, bucket)
