from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from rapidpipe.pipeline import Pipeline


# --------------------------------------------------------------------------- #
#  Execution mode                                                              #
# --------------------------------------------------------------------------- #
class ExecutionMode(Enum):
    AUTO = "auto"  # sync → thread pool; async → event loop (default)
    THREAD = "thread"  # always run in thread pool (CPU-bound, GIL-releasing)
    ASYNC = "async"  # must define aprocess(); I/O-bound coroutine
    # always run synchronously, no offloading (debug / trivial layers)
    INLINE = "inline"


# --------------------------------------------------------------------------- #
#  Exceptions                                                                  #
# --------------------------------------------------------------------------- #
class LayerComplete(Exception):
    """Raise from process() to signal this layer has no more data (e.g., end of file)."""


class PipelineStop(Exception):
    """Raise from process() to stop the entire pipeline cleanly."""


# --------------------------------------------------------------------------- #
#  Access specification                                                       #
# --------------------------------------------------------------------------- #
class AccessType(Enum):
    CURRENT = "current"
    INDEXED = "indexed"
    INDEX_RANGE = "index_range"
    TIME_RANGE = "time_range"
    PIPELINE_CURRENT = "pipeline_current"
    PIPELINE_INDEXED = "pipeline_indexed"
    PIPELINE_INDEX_RANGE = "pipeline_index_range"
    PIPELINE_TIME_RANGE = "pipeline_time_range"


@dataclass(slots=True)
class DependencyInfo:
    layer_name: str
    output_name: str
    access_type: AccessType
    index: Optional[int] = None
    index_start: Optional[int] = None
    index_end: Optional[int] = None
    time_start: Optional[float] = None
    time_end: Optional[float] = None
    is_pipeline_dependency: bool = False
    resolved_layer: Optional[str] = None

    def __str__(self) -> str:
        """Returns the dependency in the string format used for specification."""
        prefix = f"{self.layer_name}." if self.layer_name else ""
        base = f"{prefix}{self.output_name}"

        if self.access_type in (AccessType.CURRENT, AccessType.PIPELINE_CURRENT):
            return base

        if self.access_type in (AccessType.INDEXED, AccessType.PIPELINE_INDEXED):
            return f"{base}[-{self.index}]"

        if self.access_type in (AccessType.INDEX_RANGE, AccessType.PIPELINE_INDEX_RANGE):
            start = f"-{self.index_start}" if self.index_start is not None else ""
            end = f":-{self.index_end}" if self.index_end is not None else ":"
            return f"{base}[{start}{end}]"

        if self.access_type in (AccessType.TIME_RANGE, AccessType.PIPELINE_TIME_RANGE):
            start = f"-{self.time_start}s" if self.time_start is not None else ""
            end = f":-{self.time_end}s" if self.time_end is not None else ":"
            return f"{base}[{start}{end}]"

        return base

    def __repr__(self) -> str:
        if self.resolved_layer:
            return f'DependencyInfo({self.__str__()}, resolved_layer="{self.resolved_layer}")'
        return f"DependencyInfo({self.__str__()})"



class _OutputValue(str):
    def __new__(cls, layer_name: str, variable_name: str):
        value = f"{layer_name}.{variable_name}"
        obj = super().__new__(cls, value)
        obj._layer_name = layer_name
        obj._variable_name = variable_name
        return obj

    def __str__(self) -> str:
        return f"{self._layer_name}.{self._variable_name}"

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.start is None:
                raise ValueError("Slice start cannot be None")
            slice_str = f"{key.start}:{'' if key.stop is None else key.stop}"
            return f"{self._layer_name}.{self._variable_name}[{slice_str}]"
        elif isinstance(key, int):
            return f"{self._layer_name}.{self._variable_name}[-{abs(key)}]"
        return super().__getitem__(key)


class _Outputs:
    def __init__(self, name: str, *args):
        """
        Initialize the _Outputs container with a name and output fields.

        :param name: Name of the outputs container (typically the layer name).
        :param args: String names of the outputs.
        """
        self._name = name
        self._fields = list(args)

    def __getattr__(self, item: str) -> _OutputValue:
        """
        Allow accessing outputs as attributes securely.
        """
        if item in self._fields:
            return _OutputValue(self._name, item)
        raise AttributeError(
            f"Output '{item}' not defined in layer '{self._name}'")

    def __iter__(self):
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __contains__(self, item: str) -> bool:
        return item in self._fields

    def __getitem__(self, index: int) -> str:
        return self._fields[index]


# --------------------------------------------------------------------------- #
#  Layer base class                                                           #
# --------------------------------------------------------------------------- #
class Layer(ABC):
    """
    Minimal, pipeline-agnostic base class.
    Responsibilities:
        - expose name & outputs
        - parse input definitions into DependencyInfo
        - provide a process(**kwargs) -> Any|list|tuple  hook
        - optional async, conditional execution, and lifecycle hooks
    """

    # execution_mode class attribute — set once at class definition
    execution_mode: ExecutionMode = ExecutionMode.AUTO

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        inputs: Optional[Dict[str, str]] = None,
        outputs: Optional[List[str]] = None,
    ):
        """
        Initialize the generic layer base.

        Args:
            name: The optional string name of the layer (defaults to UUID if not provided).
            inputs: A dictionary mapping process() kwarg names to dependency strings.
                Example: `{"x": "other_layer.y[-1]"}`
            outputs: A list of string names for the outputs produced by this layer.
                Example: `["y", "z"]`
        """
        self._name: str = name or str(uuid4())
        self._has_name: bool = name is not None
        self._output_names: List[str] = outputs or []
        self._output_defs = _Outputs(self._name, *self._output_names)
        self._parsed_deps: Dict[str, DependencyInfo] = {}

        self._pipeline: Optional[Pipeline] = None

        # Parse input definitions
        if inputs:
            self._parse_inputs(inputs)

    # --------------------------------------------------------------------- #
    #  Public read-only API                                                  #
    # --------------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """Return the unique string identifier for this layer instance."""
        return self._name

    @property
    def display_name(self) -> str:
        """
        Returns the display name of the layer.
        If a name was provided, it returns that; otherwise, it returns the class name.
        """
        return self._name if self._has_name else self.__class__.__name__

    @property
    def outputs(self) -> List[str]:
        """Return the list of output variable names declared by this layer."""
        return self._output_defs

    @property
    def parsed_dependencies(self) -> Dict[str, DependencyInfo]:
        """Return the parsed DependencyInfo definitions for inputs."""
        return self._parsed_deps

    @property
    def effective_execution_mode(self) -> ExecutionMode:
        """
        Return the dynamically computed effective execution mode when attached to a pipeline,
        otherwise return the explicitly declared execution mode.
        """
        if self._pipeline and hasattr(self._pipeline, '_effective_execution_modes'):
            return self._pipeline._effective_execution_modes.get(self.name, self.execution_mode)
        return self.execution_mode

    # --------------------------------------------------------------------- #
    #  Pipeline integration                                                  #
    # --------------------------------------------------------------------- #
    def add_to_pipeline(self, pipeline: Pipeline) -> None:
        """
        Register this layer internally to a parent Pipeline.

        Args:
            pipeline: The Pipeline instance adding this layer.
        Raises:
            ValueError: If the layer already belongs to a pipeline.
        """
        if self._pipeline is not None:
            raise ValueError("Layer already belongs to a pipeline.")
        self._pipeline = pipeline

    # --------------------------------------------------------------------- #
    #  Processing contract                                                   #
    # --------------------------------------------------------------------- #
    @abstractmethod
    def process(self, **inputs) -> Any:
        """
        Accepts the resolved inputs keyed by the names defined in `inputs=`.
        Must return either:
            - a single value  (assigned to the first output)
            - a list/tuple    (assigned to outputs in order)
            - None            (skip)
        """
        pass

    async def aprocess(self, **inputs) -> Any:
        """
        Async process. Override this instead of process() for I/O-bound layers.
        The pipeline calls this when execution_mode is ASYNC or AUTO and this method exists.
        Default implementation wraps process() for backward compatibility.
        """
        return self.process(**inputs)

    # --------------------------------------------------------------------- #
    #  Conditional execution                                                 #
    # --------------------------------------------------------------------- #
    def should_run(self, **inputs) -> bool:
        """
        Return False to skip this layer for the current cycle.
        Inputs are the already-resolved values. Useful for conditional branches,
        rate limiting (e.g., only run every N cycles), or guarding against None inputs.
        Default: always run.
        """
        return True

    # --------------------------------------------------------------------- #
    #  Lifecycle hooks                                                       #
    # --------------------------------------------------------------------- #
    def on_start(self) -> None:
        """Called once before the pipeline's first cycle. Use for resource acquisition."""
        pass

    def on_stop(self) -> None:
        """Called once after the pipeline stops. Use for cleanup."""
        pass

    def on_complete(self) -> None:
        """Called when a producer layer raises LayerComplete. Override to react to completions."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Return a dictionary of internal state to be serialized by the pipeline.
        Override this to save custom state variables when `save_state()` is called.
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore internal state from a dictionary when `load_state()` is called.
        Override this to handle loading custom state variables.
        """
        pass

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                     #
    # --------------------------------------------------------------------- #
    def _parse_inputs(self, raw: Dict[str, str]) -> None:
        """
        Convert string dependency specifiers to DependencyInfo objects.
        """
        sig = inspect.signature(self.process)
        # accept extra keys if **kwargs exists
        has_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        allowed = set(sig.parameters)
        for key, spec in raw.items():
            if not has_kwargs and key not in allowed:
                raise TypeError(
                    f"Unknown parameter '{key}' in layer '{self.name}' "
                    f"(allowed: {allowed})"
                )
            self._parsed_deps[key] = self._parse_spec(spec)

    # ---------- single regex-based parser --------------------------------- #
    _PIPE_BASIC = re.compile(r"^([^.\[\]]+)$")
    _PIPE_INDEX = re.compile(r"^([^.\[\]]+)\[(.+)\]$")
    _LAYER_BASIC = re.compile(r"^([^.\[\]]+)\.([^.\[\]]+)$")
    _LAYER_INDEX = re.compile(r"^([^.\[\]]+)\.([^.\[\]]+)\[(.+)\]$")

    @classmethod
    def _parse_spec(cls, spec: str) -> DependencyInfo:
        """
        Convert a dependency string into DependencyInfo.
        Grammar:
            pipelineVar
            pipelineVar[-n]
            pipelineVar[-n:-m]
            pipelineVar[-3.5s:-1s]
            layerName.var
            layerName.var[-n]
            layerName.var[-n:-m]
            layerName.var[-3.5s:-1s]
        """
        spec = spec.strip()

        # 1) Pipeline-level current
        m = cls._PIPE_BASIC.match(spec)
        if m:
            return DependencyInfo(
                "", m.group(1), AccessType.PIPELINE_CURRENT, is_pipeline_dependency=True
            )

        # 2) Pipeline-level indexed/ranged
        m = cls._PIPE_INDEX.match(spec)
        if m:
            name, idx = m.groups()
            return cls._parse_index_part(name, idx, is_pipeline=True)

        # 3) Layer-level current
        m = cls._LAYER_BASIC.match(spec)
        if m:
            layer, var = m.groups()
            return DependencyInfo(layer, var, AccessType.CURRENT)

        # 4) Layer-level indexed/ranged
        m = cls._LAYER_INDEX.match(spec)
        if m:
            layer, var, idx = m.groups()
            return cls._parse_index_part(layer + "." + var, idx, is_pipeline=False)

        raise ValueError(f"Invalid dependency specification: {spec}")

    # ---------- helper for index parsing ---------------------------------- #
    _INDEX_RANGE = re.compile(r"-(\d+)(?::(-?\d*))?")
    _TIME_RANGE = re.compile(r"-(\d+(?:\.\d+)?)s(?::(-?\d+(?:\.\d+)?)s)?")

    @classmethod
    def _parse_index_part(
        cls,
        base: str,
        idx_spec: str,
        *,
        is_pipeline: bool,
    ) -> DependencyInfo:
        """
        Parse the bracketed portion of a dependency string to extract time or index ranges.

        Args:
            base: The base variable name or layer.var string.
            idx_spec: The string inside the brackets (e.g. "-1", "-2:-1", "-3s:-1s").
            is_pipeline: Boolean flag indicating if this is a pipeline-level dependency.

        Returns:
            A constructed DependencyInfo object.

        Raises:
            ValueError: If the index specifier inside the brackets is malformed.
        """
        idx_spec = idx_spec.replace("sec", "s")

        # time range
        m = cls._TIME_RANGE.match(idx_spec)
        if m:
            start, end = m.groups()
            return DependencyInfo(
                "" if is_pipeline else base.split(".")[0],
                base if is_pipeline else base.split(".", 1)[1],
                AccessType.PIPELINE_TIME_RANGE
                if is_pipeline
                else AccessType.TIME_RANGE,
                time_start=float(start),
                time_end=abs(float(end)) if end else None,
                is_pipeline_dependency=is_pipeline,
            )

        # index / index range
        m = cls._INDEX_RANGE.match(idx_spec)
        if m:
            start, end = m.groups()
            if ":" not in idx_spec:
                # single index
                return DependencyInfo(
                    "" if is_pipeline else base.split(".")[0],
                    base if is_pipeline else base.split(".", 1)[1],
                    AccessType.PIPELINE_INDEXED if is_pipeline else AccessType.INDEXED,
                    index=int(start),
                    is_pipeline_dependency=is_pipeline,
                )
            else:
                # index range (end may be None or empty string)
                return DependencyInfo(
                    "" if is_pipeline else base.split(".")[0],
                    base if is_pipeline else base.split(".", 1)[1],
                    AccessType.PIPELINE_INDEX_RANGE
                    if is_pipeline
                    else AccessType.INDEX_RANGE,
                    index_start=int(start),
                    index_end=abs(int(end)) if end else None,
                    is_pipeline_dependency=is_pipeline,
                )

        raise ValueError(f"Invalid index specifier: {idx_spec}")

    # --------------------------------------------------------------------- #
    #  Nice string / repr                                                    #
    # --------------------------------------------------------------------- #
    def __repr__(self) -> str:
        """Return a string representation suitable for debugging."""
        return (
            f"{self.__class__.__name__}"
            f"(name={self._name!r}, inputs={list(self._parsed_deps)}, outputs={self._output_names})"
        )

    def __str__(self) -> str:
        """Return a human-readable summary of the layer's inputs and outputs."""
        lines = [f"{self.__class__.__name__}({self._name})"]
        if self._parsed_deps:
            lines.append("  inputs:")
            lines.extend(f"    {k}: {v}" for k, v in self._parsed_deps.items())
        if self._output_names:
            lines.append("  outputs:")
            lines.extend(f"    {i}: {o}" for i,
                         o in enumerate(self._output_names))
        return "\n".join(lines)
