# Owner(s): ["module: dsl-native-ops"]
#
# Host-only tests for the static-fragment datapath's THREAD MAP. TileMap is plain arithmetic --
# no kernel, no GPU -- and it is where a fold order describes itself, so the properties an order
# depends on are checkable here rather than only through a compiled kernel two commits later.

import math

from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@skipIfNoCuteDSL
class TestTileDatapath(TestCase):
    def _tm(self, **kw):
        from torch._native.ops.reductions import tile

        return tile.TileMap(**kw)

    def test_rejects_a_partial_warp(self):
        # tpr must be 1 or a whole number of warps: the lane merge shuffles across a full warp, so
        # a partial one silently folds the wrong lanes.
        with self.assertRaises(ValueError):
            self._tm(N=512, itemsize=4, tpr=48, loads=1)

    def test_exact_is_derived_and_overridable(self):
        # `exact` means the tile covers the row with nothing left over, which is what lets the load
        # emit no predication at all. It is derived from N by default; a BATCHED tile covers only
        # its batch, so that caller passes exact=False and must keep its bound checks.
        covering = self._tm(N=256, itemsize=4, tpr=32, loads=2)  # 4 * 2 * 32 == 256
        self.assertTrue(covering.exact)
        ragged = self._tm(N=252, itemsize=4, tpr=32, loads=2)
        self.assertFalse(ragged.exact)
        self.assertFalse(
            self._tm(N=256, itemsize=4, tpr=32, loads=2, exact=False).exact,
            "an explicit exact=False must win over the derivation",
        )

    def test_vec_override_is_what_an_order_needs(self):
        # By default vec is gcd-derived from N, which changes with N and so would change an
        # order's add DAG. An order defines itself as 16 // itemsize regardless of N and pads the
        # ragged tail with identities, so the override has to be honoured verbatim.
        derived = self._tm(N=252, itemsize=4, tpr=32, loads=2)
        self.assertEqual(derived.vec, math.gcd(252, 4))
        fixed = self._tm(N=252, itemsize=4, tpr=32, loads=2, vec=4, exact=False)
        self.assertEqual(fixed.vec, 4)

    def test_strides_are_the_only_difference_between_the_two_orders(self):
        # The regular and inner-tree orders read the SAME elements into the SAME registers and
        # differ only in which chunk goes to which warp -- i.e. in the l/w strides swapping. Both
        # must keep stride 1 innermost (that is what makes the load coalesce) and both must be a
        # permutation of the same column set.
        kw = dict(N=1024, itemsize=4, tpr=128, loads=2)
        row_major = self._tm(**kw)
        warp_major = self._tm(**kw, warp_major=True)
        self.assertNotEqual(row_major.strides(), warp_major.strides())
        for tm in (row_major, warp_major):
            cols = {
                tm.col_base(lane, w, l)
                for lane in range(32)
                for w in range(tm.nw)
                for l in range(tm.loads)
            }
            self.assertEqual(
                len(cols), 32 * tm.nw * tm.loads, "two threads share a chunk"
            )
            self.assertEqual(min(cols), 0)
            self.assertEqual(max(cols), kw["N"] - tm.vec)

    def test_align_bytes_tracks_the_load_width(self):
        # The declared alignment is what makes the DSL emit the wide instruction, and declaring
        # more than the layout proves faults at launch. With a DERIVED vec it always divides N, so
        # the wide load is legal and the alignment is vec * itemsize. The one case where it is not
        # is an ORDER's explicit vec, which is 16 // itemsize regardless of N: there the load falls
        # back to per-element reads and the declaration has to fall back with it.
        wide = self._tm(N=1024, itemsize=4, tpr=32, loads=1)
        self.assertTrue(wide.wide_ok)
        self.assertEqual(wide.align_bytes(4), wide.vec * 4)

        padded = self._tm(N=253, itemsize=4, tpr=32, loads=2, vec=4, exact=False)
        self.assertFalse(padded.wide_ok, "253 is not a multiple of the order's vec")
        self.assertEqual(padded.align_bytes(4), 4)


if __name__ == "__main__":
    run_tests()
