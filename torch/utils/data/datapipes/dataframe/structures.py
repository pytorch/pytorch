from torch.utils.data import (
    DataChunk,
)

class DataChunkDF(DataChunk):
    """
        DataChunkDF iterating over individual items inside of DataFrame containers,
        to access DataFrames user `raw_iterator`
    """

    def __iter__(self):
        for df in self.items:
            for record in df.to_records(index=False):
                yield record

    def __len__(self):
        total_len = 0
        for df in self.items:
            total_len += len(df)
        return total_len
