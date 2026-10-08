# mypy: allow-untyped-defs

from .flex_attn_fwd_helpers import flyc, fx, i32, u32, unroll


@flyc.jit
def make_paired_metadata_ops(lane, load_i32, wave_size, seq_kv, sparse_kv_size):
    def pair_word(view, row, count, capacity, word):
        bits = u32(0)
        for group in unroll((capacity + wave_size - 1) // wave_size):
            slot = i32(group * wave_size) + lane
            safe_slot = (slot < i32(capacity)).select(slot, i32(0))
            block = load_i32(view, row * i32(capacity) + safe_slot)
            valid = (slot < count) & (block // i32(32) == i32(word))
            bit = u32(1) << (u32(block) & u32(31))
            bits = bits | valid.select(bit, u32(0))
        for step in unroll(6):
            bits = bits | bits.shuffle_xor(i32(1 << step), i32(wave_size))
        return u32(fx.rocdl.readfirstlane(i32.ir_type, i32(bits).ir_value()))

    def pair_cache(words):
        cache = []
        for group in unroll((seq_kv // sparse_kv_size + wave_size - 1) // wave_size):
            index = i32(group * wave_size) + lane
            prefix = i32(0)
            selected = i32(0)
            for word in unroll(len(words)):
                bits = words[word]
                count = i32(fx.math.ctpop(bits))
                valid = (index >= prefix) & (index < prefix + count)
                rank = valid.select(index - prefix, i32(0))
                position = i32(word * 32)
                for level in unroll(5):
                    width = 16 >> level
                    low_count = i32(fx.math.ctpop(bits & u32((1 << width) - 1)))
                    upper = rank >= low_count
                    rank = upper.select(rank - low_count, rank)
                    bits = upper.select(bits >> u32(width), bits)
                    position = position + upper.select(i32(width), i32(0))
                selected = valid.select(position, selected)
                prefix = prefix + count
            cache.append(selected)
        return cache

    def read_pair_cache(cache, index):
        source_lane = index % i32(wave_size)
        selected = i32(
            fx.rocdl.readlane(i32.ir_type, cache[0].ir_value(), source_lane.ir_value())
        )
        for group in unroll(1, (seq_kv // sparse_kv_size + wave_size - 1) // wave_size):
            value = i32(
                fx.rocdl.readlane(
                    i32.ir_type, cache[group].ir_value(), source_lane.ir_value()
                )
            )
            selected = (index >= i32(group * wave_size)).select(value, selected)
        return selected

    def merge_rows(
        mask_row,
        full_count,
        second_full_count,
        partial_count,
        second_partial_count,
        full_indices,
        partial_indices,
        full_capacity,
        partial_capacity,
    ):
        full_bits = []
        partial_bits = []
        paired_full_count = i32(0)
        paired_partial_count = i32(0)
        for word in unroll((seq_kv // sparse_kv_size + 31) // 32):
            first_full_word, second_full_word = [
                pair_word(
                    full_indices,
                    mask_row + i32(row),
                    count,
                    full_capacity,
                    word,
                )
                for row, count in enumerate((full_count, second_full_count))
            ]
            first_partial_word, second_partial_word = [
                pair_word(
                    partial_indices,
                    mask_row + i32(row),
                    count,
                    partial_capacity,
                    word,
                )
                for row, count in enumerate((partial_count, second_partial_count))
            ]
            common = first_full_word & second_full_word
            partial = (
                first_full_word
                | second_full_word
                | first_partial_word
                | second_partial_word
            ) & ~common
            full_bits.append(common)
            partial_bits.append(partial)
            paired_full_count = paired_full_count + i32(fx.math.ctpop(common))
            paired_partial_count = paired_partial_count + i32(fx.math.ctpop(partial))
        full_count = paired_full_count
        partial_count = paired_partial_count
        pair_full_cache = pair_cache(full_bits)
        return full_count, partial_count, pair_full_cache, partial_bits

    return pair_word, pair_cache, read_pair_cache, merge_rows
