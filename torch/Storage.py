class RealStorage(RealStorageBase):
    def __str__(self):
        content = ' ' + '\n '.join(str(self[i]) for i in range(len(self)))
        return content + '\n[torch.RealStorage of size {}]'.format(len(self))

    def __repr__(self):
        return str(self)

