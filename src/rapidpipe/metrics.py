from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LayerMetrics:
    call_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    # Store times in nanoseconds for higher resolution
    wall_times_ns: deque = field(default_factory=lambda: deque(maxlen=500))
    output_shapes: Dict[str, deque] = field(default_factory=dict)

    @property
    def history_ms(self) -> list[float]:
        """Convert recorded nanosecond wall times to milliseconds."""
        return [t / 1_000_000.0 for t in self.wall_times_ns]

    @property
    def mean_ms(self) -> float:
        """Calculate the average execution time in milliseconds."""
        return statistics.mean(self.history_ms) if self.wall_times_ns else 0.0

    @property
    def p99_ms(self) -> float:
        """Calculate the 99th percentile execution time in milliseconds."""
        if not self.wall_times_ns:
            return 0.0
        h = sorted(self.history_ms)
        return h[int(len(h) * 0.99)]

    @property
    def p50_ms(self) -> float:
        """Calculate the median (50th percentile) execution time in milliseconds."""
        return statistics.median(self.history_ms) if self.wall_times_ns else 0.0


class MetricsCollector:
    def __init__(self, window: int = 500, track_shapes: bool = True):
        """
        Initialize a new metrics collector for a pipeline.
        
        Args:
            window: Number of recent cycles to retain in the metrics history deque.
            track_shapes: Whether to track the shapes/sizes of output arrays/lists.
        """
        self._window = window
        self._track_shapes = track_shapes
        self._metrics: Dict[str, LayerMetrics] = {}

    def register(self, layer_name: str) -> None:
        """
        Register a new layer in the metrics tracker, initializing its history storage.
        
        Args:
            layer_name: The string identifier of the layer.
        """
        self._metrics[layer_name] = LayerMetrics(
            wall_times_ns=deque(maxlen=self._window)
        )

    @contextmanager
    def measure(self, layer_name: str):
        """
        A context manager to wrap layer execution and record execution time.
        
        Args:
            layer_name: The layer being executed.
            
        Yields:
            None. Captures exceptions and execution wall time automatically.
        """
        m = self._metrics[layer_name]
        m.call_count += 1
        t0 = time.perf_counter_ns()
        try:
            yield
        except Exception:
            m.error_count += 1
            raise
        finally:
            m.wall_times_ns.append(time.perf_counter_ns() - t0)

    def record_skip(self, layer_name: str) -> None:
        """
        Record that a layer was skipped during a cycle.
        
        Args:
            layer_name: The layer that was bypassed via should_run().
        """
        m = self._metrics[layer_name]
        m.skipped_count += 1

    def record_outputs(self, layer_name: str, outputs: Dict[str, Any]) -> None:
        """
        Inspect layer outputs and record their shapes/lengths if track_shapes is enabled.
        
        Args:
            layer_name: The producing layer.
            outputs: Dictionary of the produced outputs for this cycle.
        """
        if not self._track_shapes:
            return
        m = self._metrics[layer_name]
        for k, v in outputs.items():
            if k not in m.output_shapes:
                m.output_shapes[k] = deque(maxlen=self._window)
            shape = getattr(v, 'shape', None) or (len(v) if hasattr(v, '__len__') else None)
            if shape is not None:
                m.output_shapes[k].append(shape)

    def summary(self) -> Dict[str, Dict]:
        """
        Generate a summarized dictionary of performance metrics across all layers.
        
        Returns:
            A dictionary mapping layer names to their aggregated execution statistics.
        """
        return {
            name: {
                "calls": m.call_count,
                "errors": m.error_count,
                "skipped": m.skipped_count,
                "mean_ms": round(m.mean_ms, 3),
                "p50_ms": round(m.p50_ms, 3),
                "p99_ms": round(m.p99_ms, 3),
            }
            for name, m in self._metrics.items()
        }

    def report(self, sort_by: str = "mean_ms") -> str:
        """
        Generate a formatted, human-readable string table of the performance profile.
        
        Args:
            sort_by: The metric column to sort the table by descending order.
            
        Returns:
            A tabulated string report of layer latencies and error rates.
        """
        lines = [f"{'Layer':<30} {'Calls':>8} {'Mean ms':>10} {'P50 ms':>10} {'P99 ms':>10} {'Errors':>8}"]
        lines.append("-" * 80)
        
        items = list(self.summary().items())
        # Sort if the key exists, default 0
        items.sort(key=lambda x: x[1].get(sort_by, 0), reverse=True)
        
        for name, row in items:
            lines.append(
                f"{name:<30} {row['calls']:>8} {row['mean_ms']:>10} "
                f"{row['p50_ms']:>10} {row['p99_ms']:>10} {row['errors']:>8}"
            )
        return "\n".join(lines)

    def bottleneck(self) -> Optional[str]:
        """Return the name of the layer with the highest p99 latency."""
        if not self._metrics:
            return None
        return max(self._metrics, key=lambda n: self._metrics[n].p99_ms)
