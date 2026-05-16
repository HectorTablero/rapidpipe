"""
Example 11 — Fine-Grained Task-Graph Scheduling
=================================================
Demonstrates that the pipeline scheduler launches each layer as
soon as its dependencies are met, rather than waiting for all
co-tier layers to finish.

DAG:
    A ──► B (5 s) ──► C (fast) ──► F
    A ──► D (fast) ──► E (5 s)  ──► F

With a tier-barrier scheduler the two slow layers would run
sequentially (Tier 1 waits for B *and* D → 5 s, then Tier 2
waits for C *and* E → 5 s, total ≈ 10 s).

With the fine-grained scheduler both branches overlap, giving
a total cycle time of ≈ 5 s (the critical path length).

Uses MetricsCollector to prove it.
"""
from rapidpipe import Pipeline, Layer, MetricsCollector, ExecutionMode
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


SLOW_DURATION = 3  # seconds


# ── Layers ─────────────────────────────────────────────────────────────────── #

class Source(Layer):
    """A — root node, negligible cost."""
    # execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="A", outputs=["x"])
        self._n = 0

    def process(self) -> int:
        self._n += 1
        return self._n


class SlowB(Layer):
    """B — simulates a 5 s computation (branch 1)."""

    def __init__(self):
        super().__init__(name="B", inputs={"x": "A.x"}, outputs=["y"])

    def process(self, x) -> int:
        time.sleep(SLOW_DURATION)
        return x * 10


class FastC(Layer):
    """C — instant passthrough after B."""
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="C", inputs={"y": "B.y"}, outputs=["z"])

    def process(self, y) -> int:
        return y + 1


class FastD(Layer):
    """D — instant passthrough (branch 2)."""
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="D", inputs={"x": "A.x"}, outputs=["w"])

    def process(self, x) -> int:
        return x * 100


class SlowE(Layer):
    """E — simulates a 5 s computation (branch 2)."""

    def __init__(self):
        super().__init__(name="E", inputs={"w": "D.w"}, outputs=["v"])

    def process(self, w) -> int:
        time.sleep(SLOW_DURATION)
        return w + 1


class Sink(Layer):
    """F — joins both branches, negligible cost."""
    # execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(
            name="F",
            inputs={"z": "C.z", "v": "E.v"},
            outputs=["result"],
        )

    def process(self, z, v) -> int:
        return z + v


# ── Build pipeline ─────────────────────────────────────────────────────────── #

metrics = MetricsCollector()
pipe = Pipeline(
    Source(),
    SlowB(),
    FastC(),
    FastD(),
    SlowE(),
    Sink(),
    metrics=metrics,
)

print("=" * 60)
print("Example 11 — Fine-Grained Task-Graph Scheduling")
print("=" * 60)
print()


# ── Execute ────────────────────────────────────────────────────────────────── #

print(f"Running 1 cycle (B and E each sleep {SLOW_DURATION}s)...")
print()

wall_start = time.perf_counter()
pipe.run_sequence(1)
wall_elapsed = time.perf_counter() - wall_start

# ── Results ────────────────────────────────────────────────────────────────── #

result = pipe.outputs.result
# A produces 1, branch 1: B→10, C→11.  Branch 2: D→100, E→101.  F→112.
expected = (1 * 10 + 1) + (1 * 100 + 1)  # 11 + 101 = 112
assert result == expected, f"Expected {expected}, got {result}"

print(f"Result:   F = C + E = {pipe.outputs.z} + {pipe.outputs.v} = {result}")
print()

# Profiling report
print(metrics.report())

# ── Timing proof ───────────────────────────────────────────────────────────── #

print(f"Wall-clock time: {wall_elapsed:.2f}s")
print()

if wall_elapsed < SLOW_DURATION * 1.5:
    print(
        f"[OK] Both branches overlapped -- total ~{SLOW_DURATION}s, not {SLOW_DURATION * 2}s")
    print("     The fine-grained scheduler eliminated the tier-barrier penalty.")
else:
    print(
        f"[!!] Took {wall_elapsed:.2f}s -- branches did NOT overlap as expected.")

print()

# Compare against hypothetical tier-barrier time
tier_barrier_time = SLOW_DURATION * 2
speedup = tier_barrier_time / wall_elapsed
print(f"Tier-barrier estimate:  {tier_barrier_time}s")
print(f"Actual wall time:       {wall_elapsed:.2f}s")
print(f"Speedup:                {speedup:.2f}x")

print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("11 — Fine-Grained Scheduling")
# print(pipe.to_mermaid(show_metrics=True, show_class_names=False, show_legend=False))
