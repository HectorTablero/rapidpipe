from __future__ import annotations

import pytest

from rapidpipe.layer import AccessType, DependencyInfo, Layer, ExecutionMode, LayerComplete, PipelineStop


class FlexibleLayer(Layer):
    def process(self, **inputs):
        return None


class StrictLayer(Layer):
    def process(self, source):
        return source


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (
            "source.value",
            DependencyInfo("source", "value", AccessType.CURRENT),
        ),
        (
            "pipeline_value",
            DependencyInfo(
                "",
                "pipeline_value",
                AccessType.PIPELINE_CURRENT,
                is_pipeline_dependency=True,
            ),
        ),
        (
            "source.value[-2]",
            DependencyInfo("source", "value", AccessType.INDEXED, index=2),
        ),
        (
            "pipeline_value[-3]",
            DependencyInfo(
                "",
                "pipeline_value",
                AccessType.PIPELINE_INDEXED,
                index=3,
                is_pipeline_dependency=True,
            ),
        ),
        (
            "source.value[-4:-1]",
            DependencyInfo(
                "source",
                "value",
                AccessType.INDEX_RANGE,
                index_start=4,
                index_end=1,
            ),
        ),
        (
            "pipeline_value[-2.5s:-0.5s]",
            DependencyInfo(
                "",
                "pipeline_value",
                AccessType.PIPELINE_TIME_RANGE,
                time_start=2.5,
                time_end=0.5,
                is_pipeline_dependency=True,
            ),
        ),
    ],
)
def test_parse_spec_supports_current_index_and_time_forms(spec, expected):
    assert Layer._parse_spec(spec) == expected


def test_parse_inputs_rejects_unknown_parameter_without_kwargs():
    with pytest.raises(TypeError, match="Unknown parameter 'unexpected'"):
        StrictLayer(
            name="strict",
            inputs={"unexpected": "source.value"},
            outputs=["result"],
        )


def test_parse_inputs_allows_extra_keys_when_kwargs_are_present():
    layer = FlexibleLayer(
        name="flex",
        inputs={
            "first": "source.value",
            "second": "pipeline_total",
        },
        outputs=["result"],
    )

    assert set(layer.parsed_dependencies) == {"first", "second"}
    assert layer.parsed_dependencies["first"] == DependencyInfo(
        "source", "value", AccessType.CURRENT
    )
    assert layer.parsed_dependencies["second"] == DependencyInfo(
        "",
        "pipeline_total",
        AccessType.PIPELINE_CURRENT,
        is_pipeline_dependency=True,
    )


def test_outputs_and_display_name_helpers():
    named = FlexibleLayer(name="demo", outputs=["left", "right"])
    unnamed = FlexibleLayer(outputs=["value"])

    assert named.display_name == "demo"
    assert unnamed.display_name == "FlexibleLayer"
    assert named.outputs.left == "demo.left"
    assert str(named.outputs.right) == "demo.right"
    assert named.outputs.left[-3] == "demo.left[-3]"
    assert named.outputs.left[2:5] == "demo.left[2:5]"
    assert named.outputs[0] == "left"

    with pytest.raises(ValueError):
        _ = named.outputs.left[:]


def test_execution_mode_defaults_to_auto():
    layer = FlexibleLayer(name="test", outputs=["x"])
    assert layer.execution_mode == ExecutionMode.AUTO


def test_should_run_defaults_to_true():
    layer = FlexibleLayer(name="test", outputs=["x"])
    assert layer.should_run() is True


def test_lifecycle_hooks_are_callable():
    layer = FlexibleLayer(name="test", outputs=["x"])
    # Should not raise
    layer.on_start()
    layer.on_stop()


def test_exceptions_exist():
    """Verify LayerComplete and PipelineStop are importable and distinct."""
    assert LayerComplete is not PipelineStop
    with pytest.raises(LayerComplete):
        raise LayerComplete()
    with pytest.raises(PipelineStop):
        raise PipelineStop()


def test_dependency_info_repr():
    from rapidpipe.layer import DependencyInfo, AccessType
    dep1 = DependencyInfo("layer_a", "val", AccessType.CURRENT)
    assert repr(dep1) == "DependencyInfo(layer_a.val, current)"

    dep2 = DependencyInfo(
        "", "val", AccessType.PIPELINE_CURRENT, is_pipeline_dependency=True)
    assert repr(dep2) == "DependencyInfo(pipeline.val, pipeline_current)"


def test_output_value_fallback_getitem():
    from rapidpipe.layer import _OutputValue
    out = _OutputValue("layer", "var")
    with pytest.raises(TypeError):
        _ = out["string_key"]  # triggers super().__getitem__(key)


def test_layer_aprocess_default():
    import asyncio

    class DummyLayer(Layer):
        def __init__(self):
            super().__init__(name="dummy", outputs=["x"])

        def process(self):
            return 42

    layer = DummyLayer()
    res = asyncio.run(layer.aprocess())
    assert res == 42


def test_invalid_dependencies():
    class InvalidLayer(Layer):
        def __init__(self, dep_str):
            super().__init__(name="inv", inputs={"x": dep_str}, outputs=["y"])

        def process(self, x): pass

    # Invalid syntax completely
    with pytest.raises(ValueError, match="Invalid dependency specification"):
        InvalidLayer("layer.var.extra")

    # Invalid index specifier (valid base syntax, but bad index)
    with pytest.raises(ValueError, match="Invalid index specifier"):
        InvalidLayer("layer.var[abc]")


def test_layer_str_and_repr():
    class StrLayer(Layer):
        def __init__(self):
            super().__init__(name="str_layer",
                             inputs={"x": "a.v"}, outputs=["y"])

        def process(self, x): pass

    layer = StrLayer()

    # Repr
    rep = repr(layer)
    assert "StrLayer" in rep
    assert "'str_layer'" in rep
    assert "'y'" in rep

    # Str
    s = str(layer)
    assert "StrLayer(str_layer)" in s
    assert "inputs:" in s
    assert "x: DependencyInfo" in s
    assert "outputs:" in s
    assert "0: y" in s
