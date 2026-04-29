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
