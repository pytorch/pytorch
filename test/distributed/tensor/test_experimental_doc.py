# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

import warnings
import pytest


class TestExperimentalModule:
    def test_experimental_module_docstring(self):
        """Test that experimental module has proper docstring."""
        import torch.distributed.tensor.experimental as experimental
        
        # Check module has docstring
        assert experimental.__doc__ is not None
        assert "experimental" in experimental.__doc__.lower()
        assert "171905" in experimental.__doc__

    def test_experimental_import_warning(self):
        """Test that importing experimental module emits a UserWarning."""
        import sys
        
        # Remove module if already imported to test fresh import
        modules_to_remove = [k for k in sys.modules if k.startswith('torch.distributed.tensor.experimental')]
        for mod in modules_to_remove:
            del sys.modules[mod]
        
        # Now import and check for warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import torch.distributed.tensor.experimental
            
            # Check that a UserWarning was issued
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) > 0
            assert "experimental" in str(user_warnings[0].message).lower()
            assert "171905" in str(user_warnings[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
