class BasicHookState:
    def __init__(self, cref, process_group):
        r"""
        A class that holds state information that is needed by the communication hook
        during the training algorithm.
        Args:
            cref (DdpTrainer): reference to the self keyword of the trainer instance
            process_group (ProcessGroup): distributed process group
        """
        self.cref = cref
        self.process_group = process_group
        self.batch_number = -1

    def get_key(self, bucket_index):
        r"""
        A method that returns an encoded key that represents the current batch and
        bucket index.
        Args:
            bucket_index (int): index of the bucket being processed in backward
        """
        return f"{self.batch_number},{bucket_index}"

    def next_batch(self):
        r"""
        A method that increments batch_number by 1.
        """
        self.batch_number += 1
