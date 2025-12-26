# Owner(s): ["module: dynamo"]
"""
Tests for SideEffects.clone() mutation isolation.

This module tests that the SideEffects.clone() method properly isolates
mutations between the original and cloned instances, which is critical
for speculation and rollback during tracing.
"""
import torch
import torch._dynamo.test_case
from torch._dynamo.side_effects import SideEffects
from torch._dynamo.source import LocalSource
from torch._dynamo.variables.base import (
    AttributeMutationExisting,
    AttributeMutationNew,
    ValueMutationExisting,
    VariableTracker,
)
from torch._dynamo.variables.constant import ConstantVariable
from torch.testing._internal.common_utils import run_tests, TestCase


class FakeOutputGraph:
    """Minimal fake OutputGraph for testing SideEffects."""

    pass


class TestSideEffectsClone(TestCase):
    """Tests that SideEffects.clone() properly isolates mutations."""

    def setUp(self):
        super().setUp()
        # Keep a reference so it doesn't get garbage collected
        self.output_graph = FakeOutputGraph()

    def _make_side_effects(self):
        """Create a SideEffects instance for testing."""
        return SideEffects(self.output_graph)

    def test_id_to_variable_isolation(self):
        """Test that adding to id_to_variable in clone doesn't affect original."""
        se = self._make_side_effects()

        # Add initial item
        obj1 = object()
        var1 = ConstantVariable(1, source=LocalSource("x"))
        var1.mutation_type = ValueMutationExisting()
        se.id_to_variable[id(obj1)] = var1

        # Clone
        se_clone = se.clone()

        # Add new item to clone
        obj2 = object()
        var2 = ConstantVariable(2, source=LocalSource("y"))
        var2.mutation_type = ValueMutationExisting()
        se_clone.id_to_variable[id(obj2)] = var2

        # Original should not have the new item
        self.assertIn(id(obj1), se.id_to_variable)
        self.assertNotIn(id(obj2), se.id_to_variable)

        # Clone should have both items
        self.assertIn(id(obj1), se_clone.id_to_variable)
        self.assertIn(id(obj2), se_clone.id_to_variable)

    def test_store_attr_mutations_inner_dict_isolation(self):
        """Test that mutating inner dicts of store_attr_mutations doesn't affect original.

        This is the critical test - store_attr_mutations is dict[VariableTracker, dict[str, VariableTracker]]
        and the inner dicts must be copied, not shared.
        """
        se = self._make_side_effects()

        # Create a variable tracker as key
        key_var = ConstantVariable("key", source=LocalSource("k"))
        key_var.mutation_type = AttributeMutationExisting()

        # Add initial mutation
        val1 = ConstantVariable("value1", source=LocalSource("v1"))
        se.store_attr_mutations[key_var] = {"attr1": val1}

        # Clone
        se_clone = se.clone()

        # Add new attribute to clone's inner dict
        val2 = ConstantVariable("value2", source=LocalSource("v2"))
        se_clone.store_attr_mutations[key_var]["attr2"] = val2

        # Original inner dict should NOT have the new attribute
        self.assertIn("attr1", se.store_attr_mutations[key_var])
        self.assertNotIn(
            "attr2",
            se.store_attr_mutations[key_var],
            "Inner dict mutation in clone affected original - copy was shallow!",
        )

        # Clone inner dict should have both attributes
        self.assertIn("attr1", se_clone.store_attr_mutations[key_var])
        self.assertIn("attr2", se_clone.store_attr_mutations[key_var])

    def test_store_attr_mutations_outer_dict_isolation(self):
        """Test that adding new keys to store_attr_mutations doesn't affect original."""
        se = self._make_side_effects()

        # Clone empty side effects
        se_clone = se.clone()

        # Add new key to clone
        key_var = ConstantVariable("key", source=LocalSource("k"))
        key_var.mutation_type = AttributeMutationExisting()
        val = ConstantVariable("value", source=LocalSource("v"))
        se_clone.store_attr_mutations[key_var] = {"attr": val}

        # Original should not have the new key
        self.assertNotIn(key_var, se.store_attr_mutations)

        # Clone should have the key
        self.assertIn(key_var, se_clone.store_attr_mutations)

    def test_keepalive_isolation(self):
        """Test that appending to keepalive in clone doesn't affect original."""
        se = self._make_side_effects()

        # Add initial item
        obj1 = object()
        se.keepalive.append(obj1)

        # Clone
        se_clone = se.clone()

        # Add new item to clone
        obj2 = object()
        se_clone.keepalive.append(obj2)

        # Original should only have obj1
        self.assertEqual(len(se.keepalive), 1)
        self.assertIn(obj1, se.keepalive)
        self.assertNotIn(obj2, se.keepalive)

        # Clone should have both
        self.assertEqual(len(se_clone.keepalive), 2)
        self.assertIn(obj1, se_clone.keepalive)
        self.assertIn(obj2, se_clone.keepalive)

    def test_shared_references_intentional(self):
        """Test that save_for_backward and tensor_hooks are intentionally shared.

        When these contain data, the references are shared between original and clone.
        This is intentional - save_for_backward and tensor_hooks represent state
        that should be shared across speculation boundaries.
        """
        se = self._make_side_effects()

        # Add data so they are non-empty (empty lists are falsy and get replaced)
        # We use a simple marker object since we don't want to set up full autograd context
        se.save_for_backward.append("marker")
        se.tensor_hooks[0] = "marker"

        # Clone
        se_clone = se.clone()

        # Verify these are the same object (shared reference)
        self.assertIs(se.save_for_backward, se_clone.save_for_backward)
        self.assertIs(se.tensor_hooks, se_clone.tensor_hooks)

    def test_prune_dead_inplace_deletion_minority(self):
        """Test that in-place deletion is used when only minority of items are dead.

        When filtering removes less than half of items, in-place deletion should
        be used to avoid allocating new dicts.
        """
        se = self._make_side_effects()

        # Add several live variables (with ValueMutationExisting)
        live_objs = []
        for i in range(10):
            obj = object()
            live_objs.append(obj)
            var = ConstantVariable(i, source=LocalSource(f"x{i}"))
            var.mutation_type = ValueMutationExisting()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)

        # Add a couple of dead variables (with AttributeMutationNew and not in live set)
        dead_objs = []
        for i in range(2):
            obj = object()
            dead_objs.append(obj)
            var = ConstantVariable(100 + i, source=LocalSource(f"dead{i}"))
            var.mutation_type = AttributeMutationNew()  # This will be pruned
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)

        # Also add store_attr_mutations for some live vars (majority live)
        for i, obj in enumerate(live_objs[:8]):
            var = se.id_to_variable[id(obj)]
            se.store_attr_mutations[var] = {"attr": ConstantVariable(f"val{i}")}

        # And for dead vars (minority dead)
        for obj in dead_objs:
            var = se.id_to_variable[id(obj)]
            se.store_attr_mutations[var] = {"attr": ConstantVariable("dead_val")}

        # Store original dict identity to check if in-place deletion was used
        original_id_to_variable = se.id_to_variable
        original_store_attr_mutations = se.store_attr_mutations

        # Create minimal mock tx for prune_dead_object_new
        class FakeTx:
            def __init__(self):
                self.stack = []
                self.symbolic_locals = {}
                self.parent = None
                self.output = self

            @property
            def backward_state(self):
                return None

        tx = FakeTx()
        se.prune_dead_object_new(tx)

        # Since dead items are minority (2 out of 12), in-place deletion should be used
        # The original dict object should be the same
        self.assertIs(se.id_to_variable, original_id_to_variable)
        # For store_attr_mutations, 2 out of 10 are dead (minority)
        self.assertIs(se.store_attr_mutations, original_store_attr_mutations)

        # Verify dead objects are removed
        for obj in dead_objs:
            self.assertNotIn(id(obj), se.id_to_variable)

        # Verify live objects are still present
        for obj in live_objs:
            self.assertIn(id(obj), se.id_to_variable)

    def test_prune_dead_store_attr_mutations_inplace_deletion_minority(self):
        """Test that in-place deletion is used for store_attr_mutations when minority are dead.

        This tests the optimization applied to store_attr_mutations filtering in
        prune_dead_object_new - same pattern as id_to_variable.
        """
        se = self._make_side_effects()

        # Add live variables (with ValueMutationExisting - not pruned)
        live_objs = []
        live_vars = []
        for i in range(8):
            obj = object()
            live_objs.append(obj)
            var = ConstantVariable(i, source=LocalSource(f"x{i}"))
            var.mutation_type = ValueMutationExisting()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)
            live_vars.append(var)
            # Add store_attr_mutations for live vars
            se.store_attr_mutations[var] = {"attr": ConstantVariable(f"val{i}")}

        # Add dead variables (with AttributeMutationNew - will be pruned if not live)
        dead_objs = []
        dead_vars = []
        for i in range(2):
            obj = object()
            dead_objs.append(obj)
            var = ConstantVariable(100 + i, source=LocalSource(f"dead{i}"))
            var.mutation_type = AttributeMutationNew()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)
            dead_vars.append(var)
            # Add store_attr_mutations for dead vars
            se.store_attr_mutations[var] = {"attr": ConstantVariable("dead_val")}

        # Verify setup: 10 entries in store_attr_mutations, 2 will be dead
        self.assertEqual(len(se.store_attr_mutations), 10)

        # Store original dict identity
        original_store_attr_mutations = se.store_attr_mutations

        class FakeTx:
            def __init__(self):
                self.stack = []
                self.symbolic_locals = {}
                self.parent = None
                self.output = self

            @property
            def backward_state(self):
                return None

        tx = FakeTx()
        se.prune_dead_object_new(tx)

        # Since dead items are minority (2 out of 10), in-place deletion should be used
        self.assertIs(
            se.store_attr_mutations,
            original_store_attr_mutations,
            "store_attr_mutations should use in-place deletion when minority are dead",
        )

        # Verify dead vars are removed from store_attr_mutations
        for var in dead_vars:
            self.assertNotIn(var, se.store_attr_mutations)

        # Verify live vars are still present
        for var in live_vars:
            self.assertIn(var, se.store_attr_mutations)

        # Verify remaining count
        self.assertEqual(len(se.store_attr_mutations), 8)

    def test_prune_dead_store_attr_mutations_new_dict_majority(self):
        """Test that new dict is created for store_attr_mutations when majority are dead.

        When filtering removes more than half of items, a new dict should be
        created for efficiency.
        """
        se = self._make_side_effects()

        # Add a few live variables
        live_objs = []
        live_vars = []
        for i in range(2):
            obj = object()
            live_objs.append(obj)
            var = ConstantVariable(i, source=LocalSource(f"x{i}"))
            var.mutation_type = ValueMutationExisting()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)
            live_vars.append(var)
            se.store_attr_mutations[var] = {"attr": ConstantVariable(f"val{i}")}

        # Add many dead variables (majority)
        dead_objs = []
        dead_vars = []
        for i in range(8):
            obj = object()
            dead_objs.append(obj)
            var = ConstantVariable(100 + i, source=LocalSource(f"dead{i}"))
            var.mutation_type = AttributeMutationNew()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)
            dead_vars.append(var)
            se.store_attr_mutations[var] = {"attr": ConstantVariable("dead_val")}

        # Verify setup: 10 entries, 8 will be dead
        self.assertEqual(len(se.store_attr_mutations), 10)

        # Store original dict identity
        original_store_attr_mutations = se.store_attr_mutations

        class FakeTx:
            def __init__(self):
                self.stack = []
                self.symbolic_locals = {}
                self.parent = None
                self.output = self

            @property
            def backward_state(self):
                return None

        tx = FakeTx()
        se.prune_dead_object_new(tx)

        # Since dead items are majority (8 out of 10), new dict should be created
        self.assertIsNot(
            se.store_attr_mutations,
            original_store_attr_mutations,
            "store_attr_mutations should create new dict when majority are dead",
        )

        # Verify dead vars are removed
        for var in dead_vars:
            self.assertNotIn(var, se.store_attr_mutations)

        # Verify live vars are still present
        for var in live_vars:
            self.assertIn(var, se.store_attr_mutations)

        # Verify remaining count
        self.assertEqual(len(se.store_attr_mutations), 2)

    def test_prune_dead_new_dict_majority(self):
        """Test that new dict is created when majority of items are dead.

        When filtering removes more than half of items, a new dict should be
        created for efficiency.
        """
        se = self._make_side_effects()

        # Add a few live variables
        live_objs = []
        for i in range(2):
            obj = object()
            live_objs.append(obj)
            var = ConstantVariable(i, source=LocalSource(f"x{i}"))
            var.mutation_type = ValueMutationExisting()
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)

        # Add many dead variables (majority)
        dead_objs = []
        for i in range(10):
            obj = object()
            dead_objs.append(obj)
            var = ConstantVariable(100 + i, source=LocalSource(f"dead{i}"))
            var.mutation_type = AttributeMutationNew()  # This will be pruned
            se.id_to_variable[id(obj)] = var
            se.keepalive.append(obj)

        # Store original dict identity
        original_id_to_variable = se.id_to_variable

        # Create minimal mock tx for prune_dead_object_new
        class FakeTx:
            def __init__(self):
                self.stack = []
                self.symbolic_locals = {}
                self.parent = None
                self.output = self

            @property
            def backward_state(self):
                return None

        tx = FakeTx()
        se.prune_dead_object_new(tx)

        # Since dead items are majority (10 out of 12), new dict should be created
        self.assertIsNot(se.id_to_variable, original_id_to_variable)

        # Verify dead objects are removed
        for obj in dead_objs:
            self.assertNotIn(id(obj), se.id_to_variable)

        # Verify live objects are still present
        for obj in live_objs:
            self.assertIn(id(obj), se.id_to_variable)


if __name__ == "__main__":
    run_tests()
