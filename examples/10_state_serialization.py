"""
Example 10 — State Serialization
=================================
Demonstrates saving and loading pipeline state (history, cycle counts)
to disk using save_state() and load_state().
"""
from rapidpipe import Pipeline, Layer
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class StateLayer(Layer):
    def __init__(self):
        super().__init__(name="counter", outputs=["n"])
        self.val = 0

    def process(self):
        self.val += 1
        return self.val

    def get_state(self):
        return {"val": self.val}

    def set_state(self, state):
        self.val = state.get("val", 0)


class HistoryLayer(Layer):
    def __init__(self):
        super().__init__(name="history_tracker", inputs={
            "window": "counter.n[-3:]"}, outputs=["sum"])

    def process(self, window):
        return sum(window) if window else 0


print("=" * 60)
print("Example 10 — State Serialization")
print("=" * 60)
print()

# Create Pipeline A and run it
pipe_a = Pipeline(StateLayer(), HistoryLayer())
print("Running Pipeline A for 5 cycles...")
pipe_a.run_sequence(5)

print(
    f"Pipeline A Cycle: {pipe_a.cycle_count}, Outputs: n={pipe_a.outputs.n}, sum={pipe_a.outputs.sum}")

# Save state
state_file = "pipeline_state.pkl"
print(f"\nSaving Pipeline A state to '{state_file}'...")
pipe_a.save_state(state_file)

# Create a fresh Pipeline B
pipe_b = Pipeline(StateLayer(), HistoryLayer())
print("\nCreated Pipeline B (fresh).")
print(f"Pipeline B Initial Cycle: {pipe_b.cycle_count}")

# Load state into Pipeline B
print(f"\nLoading state into Pipeline B...")
pipe_b.load_state(state_file)
print(f"Pipeline B Cycle after load: {pipe_b.cycle_count}")

# Run Pipeline B
print("\nRunning Pipeline B for 2 more cycles...")
pipe_b.run_sequence(2)
print(
    f"Pipeline B Final Cycle: {pipe_b.cycle_count}, Outputs: n={pipe_b.outputs.n}, sum={pipe_b.outputs.sum}")

# Cleanup
if os.path.exists(state_file):
    os.remove(state_file)

print("=" * 60)

pipe_a.show_graph("10 — State Serialization")
# print(pipe_b.to_mermaid(show_metrics=True, show_class_names=False, show_legend=False))
