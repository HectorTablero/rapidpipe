"""
Example 2 — DAG-Based Parallel Tiers
=====================================
Three independent source layers feed into a fusion layer.
The DAG scheduler places all sources in Tier 0 (parallel)
and the fusion in Tier 1. Demonstrates:

  • Automatic tier computation from dependencies
  • Independent branches running in the same tier
  • Inspecting the computed execution tiers
"""
from rapidpipe import Pipeline, Layer, ExecutionMode
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── Layers ─────────────────────────────────────────────────────────────────── #

class CameraLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="camera", outputs=["frame"])
        self._i = 0

    def process(self) -> str:
        self._i += 1
        return f"frame_{self._i}"


class DepthLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="depth", outputs=["depth_map"])
        self._i = 0

    def process(self) -> str:
        self._i += 1
        return f"depth_{self._i}"


class IMULayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(name="imu", outputs=["orientation"])
        self._i = 0

    def process(self) -> str:
        self._i += 1
        return f"orient_{self._i}"


class FusionLayer(Layer):
    """Depends on all three sources — must run after them."""
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(
            name="fusion",
            inputs={
                "frame": "camera.frame",
                "depth": "depth.depth_map",
                "orient": "imu.orientation",
            },
            outputs=["fused"],
        )

    def process(self, frame, depth, orient) -> str:
        return f"FUSED({frame}, {depth}, {orient})"


class DisplayLayer(Layer):
    execution_mode = ExecutionMode.INLINE

    def __init__(self):
        super().__init__(
            name="display",
            inputs={"data": "fusion.fused"},
            outputs=["rendered"],
        )

    def process(self, data) -> str:
        return f"[DISPLAY] {data}"


# ── Build & inspect tiers ──────────────────────────────────────────────────── #

pipe = Pipeline(
    CameraLayer(),
    DepthLayer(),
    IMULayer(),
    FusionLayer(),
    DisplayLayer(),
)

print("=" * 60)
print("Example 2 — DAG Parallelism")
print("=" * 60)

# Run a few cycles
pipe.run_sequence(3)

print(f"Cycles run:  {pipe.cycle_count}")
print(f"Fused output: {pipe.outputs.fused}")
print(f"Display:      {pipe.outputs.rendered}")
print()
print("=" * 60)

# ── Visualize ──────────────────────────────────────────────────────────────── #
pipe.show_graph("02 — DAG Parallelism")
