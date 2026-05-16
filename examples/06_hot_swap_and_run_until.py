"""
Example 6 — Hot-Swapping & run_until
=====================================
Demonstrates:

  • replace_layer() — swap a layer at runtime
  • run_until(predicate) — run until a condition is met
  • remove_layer() — dynamically remove a layer
"""
from rapidpipe import Pipeline, Layer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers ─────────────────────────────────────────────────────────────────── #

class CounterLayer(Layer):
    def __init__(self):
        super().__init__(name="counter", outputs=["n"])
        self._n = 0

    def process(self) -> int:
        self._n += 1
        return self._n


class MultiplierLayer(Layer):
    """Multiplies the counter by a factor."""

    def __init__(self, factor: int = 2, name: str = "multiplier"):
        super().__init__(
            name=name,
            inputs={"n": "counter.n"},
            outputs=["product"],
        )
        self.factor = factor

    def process(self, n) -> int:
        return n * self.factor


class LogLayer(Layer):
    def __init__(self):
        super().__init__(
            name="logger",
            inputs={"product": "multiplier.product"},
            outputs=["log"],
        )
        self.entries = []

    def process(self, product) -> str:
        entry = f"cycle {self._pipeline.cycle_count}: product={product}"
        self.entries.append(entry)
        return entry


# ── Build ──────────────────────────────────────────────────────────────────── #

print("=" * 60)
print("Example 6 — Hot-Swapping & run_until")
print("=" * 60)
print()

pipe = Pipeline(
    CounterLayer(),
    MultiplierLayer(factor=2),
    LogLayer(),
)

# ── Phase 1: Run with factor=2 until product >= 10 ────────────────────────── #

print("Phase 1: Multiplier factor = 2")
pipe.run_until(lambda p: p.outputs.product >= 10)
print(f"  Stopped at cycle {pipe.cycle_count}")
print(f"  Counter = {pipe.outputs.n}")
print(f"  Product = {pipe.outputs.product}  (n × 2)")
print(f"  Log     = {pipe.outputs.log}")
print()

# ── Phase 2: Hot-swap multiplier to factor=10 ─────────────────────────────── #

print("Phase 2: Hot-swap multiplier to factor=10")
new_mult = MultiplierLayer(factor=10, name="multiplier")
pipe.replace_layer("multiplier", new_mult)

# Run 3 more cycles
pipe.run_sequence(3)
print(f"  Ran 3 more cycles (total: {pipe.cycle_count})")
print(f"  Counter = {pipe.outputs.n}")
print(f"  Product = {pipe.outputs.product}  (n × 10)")
print(f"  Log     = {pipe.outputs.log}")
print()

# ── Phase 3: Remove the logger ────────────────────────────────────────────── #

print("Phase 3: Remove logger layer")
pipe.remove_layer("logger")
print(f"  Remaining layers: {[l.name for l in pipe.layers]}")

pipe.run_sequence(2)
print(f"  Ran 2 more cycles (total: {pipe.cycle_count})")
print(f"  Counter = {pipe.outputs.n}")
print(f"  Product = {pipe.outputs.product}")
print("=" * 60)

# ── Visualize (final state, after removal) ─────────────────────────────────── #
pipe.show_graph("06 — After Hot-Swap & Removal")
# print(pipe.to_mermaid(show_metrics=True, show_class_names=False, show_legend=False))
