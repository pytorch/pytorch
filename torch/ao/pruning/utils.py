
def gen_two_four_sparse_mask(m, k, dtype):
    def random_mask_choice(i=None):
        choices = [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
        return choices[random.randint(0, len(choices) - 1) if i is None else i]

    mask_entries = [random_mask_choice() for i in range(m * k // 4)]
    return (
        torch.tensor(mask_entries, dtype=dtype, device=DEVICE).view(m, k).contiguous()
    )
