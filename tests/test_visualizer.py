import pytest
import os
from unittest.mock import patch
from rapidpipe import Pipeline, Layer
from rapidpipe.utils.visualizer import PipelineVisualizer, visualize_pipeline


class DummyVisualizerLayer(Layer):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def process(self, **kwargs):
        pass


def test_visualizer_graph_data():
    pipe = Pipeline(
        DummyVisualizerLayer("l1", outputs=["a"]),
        DummyVisualizerLayer("l2", inputs={"val": "l1.a[-1]"}, outputs=["b"]),
        DummyVisualizerLayer(
            "l3", inputs={"val": "l1.a[-2:-1]", "pipe_val": "b"}),
        name="test_pipe"
    )

    viz = PipelineVisualizer(pipe, name="test_viz")
    graph_data = viz.graph_data

    assert "nodes" in graph_data
    assert "links" in graph_data
    assert len(graph_data["nodes"]) == 3

    # Check link generation
    links = graph_data["links"]
    assert len(links) >= 2

    # We should have a link from l1 to l2
    l1_to_l2 = next(
        (l for l in links if l["source"] == "l1" and l["target"] == "l2"), None)
    assert l1_to_l2 is not None
    assert l1_to_l2["dependency_range"] == "[-1]"

    # Check link from l1 to l3 (index range)
    l1_to_l3 = next(
        (l for l in links if l["source"] == "l1" and l["target"] == "l3"), None)
    assert l1_to_l3 is not None
    assert l1_to_l3["dependency_range"] == "[-2:-1]"

    # Check pipeline-level link (from l2 to l3 because l2 produces 'b')
    l2_to_l3 = next(
        (l for l in links if l["source"] == "l2" and l["target"] == "l3"), None)
    assert l2_to_l3 is not None


@patch("webbrowser.open")
def test_visualize_pipeline(mock_open):
    pipe = Pipeline(DummyVisualizerLayer("l1", outputs=["a"]))

    visualize_pipeline(pipe, name="test_free", free=True)

    # Check that it tried to open a file
    mock_open.assert_called_once()
    url = mock_open.call_args[0][0]
    assert url.startswith("file://")
    assert url.endswith(".html")
