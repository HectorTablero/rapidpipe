"""
Example 7 — Numba Performance Comparison
=========================================
Demonstrates how to use the `NumbaLayer` base class for heavy
numerical computation, and compares its performance against
an identical implementation in pure Python.

The pipeline runs both layers in parallel (same tier) and
records execution metrics.
"""
from rapidpipe.numba_layer import NumbaLayer
from rapidpipe import Pipeline, Layer, MetricsCollector
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Heavy computation logic ────────────────────────────────────────────────── #

def heavy_computation(data: np.ndarray, iterations: int) -> np.ndarray:
    """A computationally expensive loop."""
    result = np.zeros_like(data)
    for i in range(data.shape[0]):
        val = data[i]
        for _ in range(iterations):
            val = (val * 1.05) ** 0.99
        result[i] = val
    return result


# ── Layers ─────────────────────────────────────────────────────────────────── #

class DataSource(Layer):
    """Provides random numpy arrays for processing."""

    def __init__(self):
        super().__init__(name="source", outputs=["data"])

    def process(self) -> np.ndarray:
        return np.random.rand(10000)


class PurePythonLayer(Layer):
    """Runs the heavy computation in pure Python."""

    def __init__(self, iterations: int):
        super().__init__(
            name="python_compute",
            inputs={"data": "source.data"},
            outputs=["result"]
        )
        self.iterations = iterations

    def process(self, data: np.ndarray) -> np.ndarray:
        return heavy_computation(data, self.iterations)


class JITCompiledLayer(NumbaLayer, nopython=True, nogil=True):
    """
    Runs the exact same computation, but JIT-compiled via Numba.
    Note: The kernel must be a staticmethod.
    """

    def __init__(self, iterations: int):
        super().__init__(
            name="numba_compute",
            inputs={"data": "source.data", "wait": "python_compute.result"},
            outputs=["result"]
        )
        self.iterations = iterations

    @staticmethod
    def kernel(data: np.ndarray, iterations: int) -> np.ndarray:
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            val = data[i]
            for _ in range(iterations):
                val = (val * 1.05) ** 0.99
            result[i] = val
        return result

    def warm_up(self):
        """Pre-compile the Numba kernel with dummy data."""
        print("  [numba] Warming up JIT compiler...")
        dummy_data = np.random.rand(10)
        self._compiled_kernel(dummy_data, self.iterations)
        print("  [numba] Warm-up complete.")

    def process(self, data: np.ndarray, wait: object = None) -> np.ndarray:
        # Call the auto-compiled kernel
        return self._compiled_kernel(data, self.iterations)


class ValidationLayer(Layer):
    """Ensures both implementations produce the same results."""

    def __init__(self):
        super().__init__(
            name="validator",
            inputs={
                "py_result": "python_compute.result",
                "nb_result": "numba_compute.result",
            },
            outputs=["is_valid"]
        )

    def process(self, py_result: np.ndarray, nb_result: np.ndarray) -> bool:
        return np.allclose(py_result, nb_result)


# ── Build & Run ────────────────────────────────────────────────────────────── #

metrics = MetricsCollector(window=50)

pipe = Pipeline(
    DataSource(),
    PurePythonLayer(iterations=300),
    JITCompiledLayer(iterations=300),
    ValidationLayer(),
    metrics=metrics
)

print("=" * 60)
print("Example 7 — Numba Performance Comparison")
print("=" * 60)
print()

# Warm up Numba layer before profiling
pipe.layers[2].warm_up()

print("\nRunning pipeline for 20 cycles...")
pipe.run_sequence(20)

print("\nProfiling Report:")
print(metrics.report())

print(f"\nFinal validation check passed: {pipe.outputs.is_valid}")
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("07 — Numba Performance")
