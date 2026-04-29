import pytest
import time
from rapidpipe.metrics import MetricsCollector


def test_metrics_empty_layer():
    metrics = MetricsCollector()
    metrics.register("empty")

    # Line 26: p99 empty
    summary = metrics.summary()
    assert summary["empty"]["p99_ms"] == 0.0

    # Line 99: bottleneck empty
    empty_metrics = MetricsCollector()
    assert empty_metrics.bottleneck() is None


def test_metrics_measure_exception():
    metrics = MetricsCollector()
    metrics.register("failing")

    with pytest.raises(ValueError):
        with metrics.measure("failing"):
            raise ValueError("Test error")

    summary = metrics.summary()
    assert summary["failing"]["errors"] == 1
    assert summary["failing"]["calls"] == 1
    # Wall time is still recorded for exceptions
    assert summary["failing"]["mean_ms"] > 0.0


def test_metrics_record_skip():
    metrics = MetricsCollector()
    metrics.register("skipper")

    metrics.record_skip("skipper")

    summary = metrics.summary()
    assert summary["skipper"]["skipped"] == 1


def test_metrics_record_outputs():
    # Line 64: Not tracking shapes
    metrics_no_track = MetricsCollector(track_shapes=False)
    metrics_no_track.register("no_track")
    metrics_no_track.record_outputs("no_track", {"out": [1, 2, 3]})
    assert not metrics_no_track._metrics["no_track"].output_shapes

    # Line 71: Tracking shapes (using list len as shape)
    metrics_track = MetricsCollector(track_shapes=True)
    metrics_track.register("track")
    metrics_track.record_outputs("track", {"out": [1, 2, 3]})

    assert metrics_track._metrics["track"].output_shapes["out"][-1] == 3

    # Object with shape attribute
    class ShapedObj:
        shape = (2, 2)

    metrics_track.record_outputs("track", {"shaped": ShapedObj()})
    assert metrics_track._metrics["track"].output_shapes["shaped"][-1] == (
        2, 2)
