import pytest
import sys
from unittest.mock import patch
from rapidpipe.numba_layer import NumbaLayer


def test_numba_layer_fallback():
    # Test fallback when numba is not installed
    with patch.dict(sys.modules, {'numba': None}):
        class DummyFallbackLayer(NumbaLayer):
            @staticmethod
            def kernel(x, y):
                return x + y

        # The kernel should be just the python function
        assert DummyFallbackLayer._compiled_kernel(1, 2) == 3


def test_numba_layer_default_process():
    class DummyNumbaLayer(NumbaLayer, nopython=False):
        @staticmethod
        def kernel(x, y):
            return x * y

    layer = DummyNumbaLayer(name="dummy", outputs=["out"])

    # Process unpacks **inputs and calls kernel
    result = layer.process(x=5, y=4)
    assert result == 20


def test_numba_layer_warm_up():
    class DummyWarmUpLayer(NumbaLayer, nopython=False):
        @staticmethod
        def kernel(val):
            return val ** 2

    # warm_up should call the kernel
    DummyWarmUpLayer.warm_up(5)
    # We can't assert much unless we mock the kernel, but it shouldn't crash


def test_numba_layer_invalid_kernel_not_staticmethod():
    with pytest.raises(TypeError, match="The 'kernel' method must be declared as a @staticmethod"):
        class InvalidKernelLayer(NumbaLayer):
            def kernel(self, x):
                return x


def test_numba_layer_multiple_outputs_warning():
    with pytest.warns(UserWarning, match="kernel return type hint"):
        class MultiOutputLayer(NumbaLayer, nopython=False):
            @staticmethod
            def kernel(x: float) -> float:
                return x

        _ = MultiOutputLayer(name="multi", outputs=["out1", "out2"])


def test_numba_layer_missing_input_keyerror():
    class MissingInputLayer(NumbaLayer, nopython=False):
        @staticmethod
        def kernel(x, y):
            return x * y

    layer = MissingInputLayer(name="dummy", outputs=["out"])

    with pytest.raises(KeyError, match="Missing required input for Numba kernel: 'y'"):
        layer.process(x=5)
