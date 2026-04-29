"""
Example 4 — Conditional Execution & Lifecycle Hooks
====================================================
Demonstrates:

  • should_run() — a layer that only fires every N cycles
  • on_start() / on_stop() — resource acquisition/release hooks
  • LayerComplete — a finite data source that exhausts itself
"""
from rapidpipe import Pipeline, Layer, LayerComplete
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers ─────────────────────────────────────────────────────────────────── #

class DataSourceLayer(Layer):
    """A finite source that produces 5 values then raises LayerComplete."""

    def __init__(self):
        super().__init__(name="source", outputs=["value"])
        self._data = iter(range(1, 6))  # values: 1, 2, 3, 4, 5
        self.started = False
        self.stopped = False

    def on_start(self):
        self.started = True
        print("  [source] on_start() -> resource acquired")

    def on_stop(self):
        self.stopped = True
        print("  [source] on_stop()  -> resource released")

    def process(self) -> int:
        try:
            val = next(self._data)
            print(f"  [source] produced: {val}")
            return val
        except StopIteration:
            print("  [source] data exhausted -> raising LayerComplete")
            raise LayerComplete()


class RateLimitedLayer(Layer):
    """Only runs on odd-numbered cycles (1, 3, 5, ...)."""

    def __init__(self):
        super().__init__(
            name="rate_limited",
            inputs={"value": "source.value"},
            outputs=["processed"],
        )
        self.run_count = 0

    def should_run(self, **inputs) -> bool:
        should = self._pipeline.cycle_count % 2 == 1
        if not should:
            print(
                f"  [rate_limited] cycle {self._pipeline.cycle_count} -> SKIPPED")
        return should

    def process(self, value) -> object:
        self.run_count += 1
        result = (value or 0) * 100
        print(
            f"  [rate_limited] cycle {self._pipeline.cycle_count} -> processed: {result}")
        return result


class AlwaysRunLayer(Layer):
    """Runs every cycle, consuming the latest available value."""

    def __init__(self):
        super().__init__(
            name="consumer",
            inputs={"data": "rate_limited.processed"},
            outputs=["final"],
        )

    def process(self, data) -> str:
        return f"consumed({data})"


# ── Build & run ────────────────────────────────────────────────────────────── #

print("=" * 60)
print("Example 4 — Conditional Execution & Lifecycle Hooks")
print("=" * 60)
print()

pipe = Pipeline(
    DataSourceLayer(),
    RateLimitedLayer(),
    AlwaysRunLayer(),
)

# Run for 8 cycles — source exhausts at cycle 5, but pipeline keeps going
pipe.run_sequence(8)

print()
print(f"Total cycles:        {pipe.cycle_count}")
print(
    f"Source active?       {'no' if 'source' in pipe._inactive_layers else 'yes'}")
print(f"Rate-limited runs:   {pipe.layers[1].run_count}")
print(f"Final output:        {pipe.outputs.final}")
print()

# Verify lifecycle hooks were called
src = pipe.layers[0]
print(f"on_start() called:   {src.started}")
print(f"on_stop() called:    {src.stopped}")
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("04 — Conditional & Lifecycle")
