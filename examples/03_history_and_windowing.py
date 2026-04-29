"""
Example 3 — History Access & Index Ranges
==========================================
A signal generator feeds a moving-average layer that reads
the last 3 cycles of history. Demonstrates:

  • Indexed history access  (output[-N])
  • Index range access      (output[-N:-M])
  • How the pipeline auto-sizes history buffers
"""
from rapidpipe import Pipeline, Layer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers ─────────────────────────────────────────────────────────────────── #

class SignalLayer(Layer):
    """Emits a simple integer sequence: 10, 20, 30, ..."""

    def __init__(self):
        super().__init__(name="signal", outputs=["value"])
        self._n = 0

    def process(self) -> int:
        self._n += 10
        return self._n


class PreviousValueLayer(Layer):
    """Reads the value from 2 cycles ago (single indexed access)."""

    def __init__(self):
        super().__init__(
            name="prev",
            inputs={"lagged": "signal.value[-2]"},
            outputs=["lagged"],
        )

    def process(self, lagged) -> object:
        return lagged


class MovingAverageLayer(Layer):
    """Computes the average of the last 3 values (index range access)."""

    def __init__(self):
        super().__init__(
            name="moving_avg",
            inputs={"window": "signal.value[-3:]"},
            outputs=["average"],
        )

    def process(self, window) -> object:
        if not window:
            return None
        return round(sum(window) / len(window), 2)


# ── Build & run ────────────────────────────────────────────────────────────── #

pipe = Pipeline(
    SignalLayer(),
    PreviousValueLayer(),
    MovingAverageLayer(),
)

N_CYCLES = 6

print("=" * 60)
print("Example 3 — History Access & Index Ranges")
print("=" * 60)
print()

# Storage requirements computed automatically
print("Storage requirements (auto-computed):")
print(f"  signal.value -> {pipe.get_storage_requirement('signal', 'value')}")
print()

print(f"{'Cycle':<8} {'Signal':<10} {'Lagged(-2)':<14} {'Window(-3:)':<20} {'MovAvg':<10}")
print("-" * 60)

for i in range(N_CYCLES):
    pipe.run_sequence(1)
    sig = pipe.outputs.value
    lag = pipe.outputs.lagged
    avg = pipe.outputs.average

    # Get the window that was used
    hist = list(pipe._history["signal"]["value"])
    window_vals = [e.value for e in hist[-3:]] if hist else []

    print(f"  {pipe.cycle_count:<6} {sig:<10} {str(lag):<14} {str(window_vals):<20} {str(avg):<10}")

print()
print(f"Final signal value: {pipe.outputs.value}")
print(
    f"Final lagged value: {pipe.outputs.lagged}  (= signal from 2 cycles ago)")
print(f"Final moving avg:   {pipe.outputs.average}")
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("03 — History & Windowing")
