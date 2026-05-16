"""
Example 12 — Numba Default Process
==================================
Demonstrates how to use the `NumbaLayer` base class's default `process()`
implementation. 

By default, `NumbaLayer.process(**inputs)` inspects the `kernel` staticmethod
signature and extracts inputs in positional order so that `nopython` mode works
without keyword errors.
"""

from rapidpipe.numba_layer import NumbaLayer
from rapidpipe import Pipeline, Layer
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class VectorSource(Layer):
    """Provides a numpy array"""

    def __init__(self):
        super().__init__(name="source", outputs=["data"])

    def process(self):
        return np.array([1.0, 2.0, 3.0, 4.0])


class ThresholdConfig(Layer):
    """Provides a threshold value"""

    def __init__(self, threshold: float):
        super().__init__(name="config", outputs=["threshold"])
        self.threshold = threshold

    def process(self):
        return self.threshold


class ClipLayer(NumbaLayer):
    """
    Clips values in an array. Unpacks `my_array` and `ceiling` via default process().
    """

    def __init__(self):
        # We map incoming data and config to names matching `kernel` parameters.
        super().__init__(
            name="clipper",
            inputs={"my_array": "source.data", "ceiling": "config.threshold"},
            outputs=["result"]
        )

    @staticmethod
    def kernel(my_array: np.ndarray, ceiling: float) -> np.ndarray:
        res = np.empty_like(my_array)
        for i in range(my_array.shape[0]):
            res[i] = my_array[i] if my_array[i] < ceiling else ceiling
        return res


print("=" * 60)
print("Example 12 - Numba Default Process")
print("=" * 60)

pipe = Pipeline(
    VectorSource(),
    ThresholdConfig(2.5),
    ClipLayer()
)

print("Running pipeline...")
pipe.run_sequence(1)

print("\nResult array after clipping (threshold = 2.5):")
print(pipe.outputs.result)


# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("12 — Numba Default Process")
# print(pipe.to_mermaid(show_metrics=True, show_class_names=False, show_legend=False))
