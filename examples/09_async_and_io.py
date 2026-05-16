"""
Example 9 — Async I/O and Aprocess
===================================
Demonstrates using the ExecutionMode.ASYNC and `aprocess` for I/O bound layers.
The pipeline automatically runs these concurrently.
"""
from rapidpipe import Pipeline, Layer, ExecutionMode
import sys
import time
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Async Layers ───────────────────────────────────────────────────────────── #

class Trigger(Layer):
    """Starts a cycle."""

    def __init__(self):
        super().__init__(name="trigger", outputs=["cycle_id"])
        self.n = 0

    def process(self) -> int:
        self.n += 1
        return self.n


class SlowAPI_A(Layer):
    """Simulates a slow network request using asyncio.sleep."""
    # execution_mode = ExecutionMode.ASYNC

    def __init__(self):
        super().__init__(name="api_a", inputs={
            "id": "trigger.cycle_id"}, outputs=["response"])

    def process(self, id: int) -> str:
        pass

    async def aprocess(self, id: int) -> str:
        await asyncio.sleep(0.5)  # Simulate I/O delay
        return f"A-{id}"


class SlowAPI_B(Layer):
    """Simulates another slow network request."""
    # execution_mode = ExecutionMode.ASYNC

    def __init__(self):
        super().__init__(name="api_b", inputs={
            "id": "trigger.cycle_id"}, outputs=["response"])

    def process(self, id: int) -> str:
        pass

    async def aprocess(self, id: int) -> str:
        await asyncio.sleep(0.5)  # Simulate I/O delay
        return f"B-{id}"


class Combiner(Layer):
    """Waits for both async layers."""

    def __init__(self):
        super().__init__(
            name="combiner",
            inputs={"resp_a": "api_a.response", "resp_b": "api_b.response"},
            outputs=["result"]
        )

    def process(self, resp_a: str, resp_b: str) -> str:
        return f"{resp_a} | {resp_b}"

# ── Build & Run ────────────────────────────────────────────────────────────── #


pipe = Pipeline(
    Trigger(),
    SlowAPI_A(),
    SlowAPI_B(),
    Combiner()
)

print("=" * 60)
print("Example 9 — Async I/O and Aprocess")
print("=" * 60)
print()

# Both API layers run in parallel in Tier 1.
# Total cycle time should be ~0.5s, not 1.0s.

start_time = time.time()
pipe.run_sequence(3)
duration = time.time() - start_time

print(f"Completed 3 cycles in {duration:.2f} seconds.")
print("If layers were purely sequential, this would take ~3.0s.")
print("Because API A and B run concurrently, each cycle takes ~0.5s.")
print(f"\nFinal combined result: {pipe.outputs.result}")
print("=" * 60)

pipe.show_graph("09 — Async IO")
# print(pipe.to_mermaid(show_metrics=True, show_class_names=False, show_legend=False))
