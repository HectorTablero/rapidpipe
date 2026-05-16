"""
Example 08: Pipeline as a Layer
================================
Demonstrating how `Pipeline` objects can be used as `Layer`s within larger pipelines:
- Sub-pipelines accepting external inputs from parents.
- Sub-pipelines returning exported outputs back to parents.
- Correct propagation of `on_start()` and `on_stop()` lifecycle hooks.
- Nested state serialization.
"""

import os
from rapidpipe import Pipeline, Layer


class DataSource(Layer):
    """Generates continuous numeric values."""

    def __init__(self):
        super().__init__(name="source", outputs=["tick"])
        self.count = 0

    def on_start(self):
        print("[DataSource] on_start() invoked")

    def process(self):
        self.count += 1
        print(f"  [DataSource] process returning {self.count}")
        return self.count

    def get_state(self) -> dict:
        return {"count": self.count}

    def set_state(self, state: dict):
        self.count = state.get("count", 0)


class Multiplier(Layer):
    """Multiplies internal data by an externally supplied factor."""

    def __init__(self):
        # Depending on "tick" (from within sub-pipeline) and "factor" (via input map)
        super().__init__(
            name="multiplier",
            inputs={"val": "source.tick", "scale": "factor"},
            outputs=["result"]
        )

    def on_start(self):
        print(f"[{self.name}] on_start() invoked")

    def process(self, val: int, scale: int) -> int:
        res = val * scale
        print(
            f"  [Multiplier] process received val={val}, scale={scale} -> returned {res}")
        return res


class PrintSink(Layer):
    """Takes result from the sub-pipeline and prints it out."""

    def __init__(self):
        super().__init__(name="sink", inputs={
            "x": "sub_pipe.result"}, outputs=[])

    def process(self, x: int):
        print(f"Sink received nested output: {x}")


print("=" * 60)
print("Example 08 - Pipeline as a Layer")
print("=" * 60)

# Build a sub-pipeline
# It requires "factor" from the outside world, and exports "multiplier.result".
sub_pipeline = Pipeline(
    DataSource(),
    Multiplier(),
    name="sub_pipe",
    inputs={"factor": "external_factor"},
    exports=["result"]
)


class FactorProvider(Layer):
    """Provides the external factor to the sub-pipeline"""

    def __init__(self):
        super().__init__(name="factor_provider", outputs=["external_factor"])

    def process(self):
        print(f"  [FactorProvider] returning 10")
        return 10


# Build Parent
parent_pipeline = Pipeline(
    FactorProvider(),
    sub_pipeline,
    PrintSink(),
    name="parent_pipe"
)

# Run cycles
print(">>> Running cycles:")
parent_pipeline.run_sequence(3)

print("\n>>> Testing Nested Serialization:")
state_path = "nested_pipe_state.pkl"
parent_pipeline.save_state(state_path)
print("Saved Parent State to file.")

print("Running 2 more cycles...")
parent_pipeline.run_sequence(2)

print("Restoring State...")
parent_pipeline.load_state(state_path)
print("Running 1 cycle post-restore (should be cycle 4):")
parent_pipeline.run_sequence(1)

print("\n>>> Shutting Down:")
parent_pipeline.stop()

# Cleanup
if os.path.exists(state_path):
    os.remove(state_path)


# ── Visualize ──────────────────────────────────────────────────────────────── #
parent_pipeline.show_graph("08 — Pipeline as Layer")
# print(parent_pipeline.to_mermaid(show_metrics=True, show_class_names=True, show_legend=False))
