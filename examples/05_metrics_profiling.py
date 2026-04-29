"""
Example 5 — Metrics & Profiling
================================
Attaches a MetricsCollector to a pipeline and inspects
per-layer timing, call counts, and identifies the bottleneck.
Demonstrates:

  • MetricsCollector integration
  • Per-layer profiling (mean, p50, p99)
  • Bottleneck identification
  • The formatted report table
"""
from rapidpipe import Pipeline, Layer, MetricsCollector, ExecutionMode
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers with varying execution times ────────────────────────────────────── #

class FastLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="fast_producer", outputs=["data"])
        self._n = 0

    def process(self) -> int:
        self._n += 1
        return self._n


class MediumLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(
            name="medium_transform",
            inputs={"data": "fast_producer.data"},
            outputs=["transformed"],
        )

    def process(self, data) -> float:
        # Simulate moderate work
        total = sum(range(5000))
        return data * 2.5


class SlowLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(
            name="slow_aggregator",
            inputs={"val": "medium_transform.transformed"},
            outputs=["result"],
        )

    def process(self, val) -> str:
        # Simulate heavier work
        total = sum(range(50000))
        return f"result={val}"


# ── Build with metrics ─────────────────────────────────────────────────────── #

metrics = MetricsCollector(window=200)

pipe = Pipeline(
    FastLayer(),
    MediumLayer(),
    SlowLayer(),
    metrics=metrics,
)

print("=" * 60)
print("Example 5 — Metrics & Profiling")
print("=" * 60)
print()

# Run enough cycles for meaningful statistics
N = 100
pipe.run_sequence(N)

# ── Display metrics ────────────────────────────────────────────────────────── #

print(f"Ran {N} cycles\n")
print("Profiling Report:")
print(metrics.report())
print()

# Summary dict
summary = metrics.summary()
for name, stats in summary.items():
    print(f"  {name}:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    print()

print(f"Bottleneck layer: {metrics.bottleneck()}")
print(f"Final output: {pipe.outputs.result}")
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("05 — Metrics Profiling")
