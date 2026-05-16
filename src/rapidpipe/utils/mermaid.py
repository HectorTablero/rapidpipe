from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional, Any
from rapidpipe.layer import AccessType, ExecutionMode

if TYPE_CHECKING:
    from rapidpipe.pipeline import Pipeline


class MermaidVisualizer:
    """
    Utility to generate Mermaid.js diagram code for a RapidPipe Pipeline.
    Supports coloring by execution mode and configurable metadata display.
    """

    # Modern, high-contrast color palette matching the web visualizer
    MODE_STYLES = {
        "AUTO": "fill:#64748b,color:#fff,stroke:#334155,stroke-width:2px",
        "ASYNC": "fill:#3b82f6,color:#fff,stroke:#1d4ed8,stroke-width:2px",
        "THREAD": "fill:#f59e0b,color:#fff,stroke:#b45309,stroke-width:2px",
        "INLINE": "fill:#10b981,color:#fff,stroke:#047857,stroke-width:2px",
    }

    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def generate(
        self,
        orientation: str = "LR",
        show_legend: bool = True,
        show_class_names: bool = True,
        show_metrics: bool = False,
        show_dependency_labels: bool = True,
        expand_subpipelines: bool = False,
    ) -> str:
        """
        Generate Mermaid code for the pipeline.

        Args:
            orientation: Mermaid graph orientation (e.g., "LR", "TB", "BT", "RL").
            show_legend: Whether to include a color legend for execution modes.
            show_class_names: Whether to show the Python class name under the instance name.
            show_metrics: Whether to include latency metrics if available.
            show_dependency_labels: Whether to label edges with the specific input/output names.
            expand_subpipelines: Whether to render nested pipelines as subgraphs.
        """
        self.show_class_names = show_class_names
        self.show_metrics = show_metrics
        self.show_dependency_labels = show_dependency_labels
        self.expand_subpipelines = expand_subpipelines

        lines = [f"graph {orientation}"]

        # 1. Add Legend (if requested)
        if show_legend:
            lines.append("    subgraph Legend [Execution Modes]")
            lines.append("        direction TB")
            lines.append("        L_AUTO(AUTO):::style_AUTO")
            lines.append("        L_ASYNC(ASYNC):::style_ASYNC")
            lines.append("        L_THREAD(THREAD):::style_THREAD")
            lines.append("        L_INLINE(INLINE):::style_INLINE")
            lines.append("    end")

        # 2. Recursively Define Nodes and Subgraphs
        node_lines, self.all_links = self._render_pipeline(self.pipeline)
        lines.extend(node_lines)

        # 3. Define Edges (Dependencies)
        for link in self.all_links:
            edge_label = ""
            if show_dependency_labels:
                dep_info = link["dependency_range"]
                edge_label = f'"{link["source_output"]} -> {link["target_input"]}{dep_info}"'
            
            connector = f" -- {edge_label} --> " if edge_label else " --> "
            lines.append(f'    {link["source"]}{connector}{link["target"]}')

        # 4. Add Class Definitions
        for mode, style in self.MODE_STYLES.items():
            lines.append(f"    classDef style_{mode} {style}")

        return "\n".join(lines)

    def _render_pipeline(self, pipeline: "Pipeline", prefix: str = "", depth: int = 1) -> Tuple[List[str], List[Dict]]:
        """Recursively render a pipeline and its layers."""
        from rapidpipe.pipeline import Pipeline
        lines = []
        all_links = []
        indent = "    " * depth

        # Collect layers and subgraphs
        for layer in pipeline.layers:
            # Use a unique ID for each node to avoid collisions in nested graphs
            # We sanitize the ID to be a valid Mermaid identifier
            node_id = f"{prefix}{layer.name}".replace(".", "_")
            
            if self.expand_subpipelines and isinstance(layer, Pipeline):
                lines.append(f"{indent}subgraph {node_id} [<b>{layer.display_name}</b>]")
                sub_lines, sub_links = self._render_pipeline(layer, prefix=f"{node_id}_", depth=depth + 1)
                lines.extend(sub_lines)
                all_links.extend(sub_links)
                lines.append(f"{indent}end")
            else:
                label = self._build_node_label(layer, self.show_class_names, self.show_metrics)
                mode = self._get_effective_mode_name(layer, pipeline)
                label = label.replace('"', '&quot;')
                lines.append(f'{indent}{node_id}["{label}"]:::style_{mode}')

        # Extract links for the current pipeline level
        current_links = self._extract_links(pipeline, prefix)
        all_links.extend(current_links)

        return lines, all_links

    def _get_effective_mode_name(self, layer, pipeline: "Pipeline") -> str:
        """Get the name of the execution mode, preferring effective mode if pipeline is built."""
        if hasattr(pipeline, "_effective_execution_modes"):
            mode = pipeline._effective_execution_modes.get(layer.name, layer.execution_mode)
        else:
            mode = layer.execution_mode
        return mode.name if hasattr(mode, "name") else str(mode).split(".")[-1]

    def _extract_links(self, pipeline: "Pipeline", prefix: str) -> List[Dict[str, Any]]:
        """Resolve all layer-to-layer links for a specific pipeline level."""
        links = []
        layers = pipeline.layers
        
        # Determine current pipeline ID for bridging links
        pipeline_id = prefix.rstrip("_")

        def find_provider(output_name, upto_idx, access_type):
            is_current = access_type in (AccessType.CURRENT, AccessType.PIPELINE_CURRENT)
            search_range = range(upto_idx - 1, -1, -1) if is_current else range(len(layers) - 1, -1, -1)
            for i in search_range:
                if output_name in layers[i].outputs:
                    return layers[i].name
            return None

        for idx, layer in enumerate(layers):
            for target_input, dep in layer.parsed_dependencies.items():
                source_layer_name = dep.layer_name
                
                # 1. Resolve pipeline-level dependencies
                if dep.is_pipeline_dependency:
                    source_layer_name = find_provider(dep.output_name, idx, dep.access_type)

                if not source_layer_name:
                    # This might be an external input to a sub-pipeline.
                    # For visualization, we could bridge it from the parent, 
                    # but Mermaid handles subgraph boundary links if we point to 
                    # the special 'PIPELINE' node or just the subgraph ID.
                    # However, to keep it simple and correct, we'll point it 
                    # to the subgraph boundary if possible.
                    if dep.is_pipeline_dependency and pipeline_id:
                        # Link from the subgraph boundary (pipeline_id) to the internal node
                        source_id = pipeline_id
                    else:
                        continue
                else:
                    source_id = f"{prefix}{source_layer_name}".replace(".", "_")

                target_id = f"{prefix}{layer.name}".replace(".", "_")
                
                # Format history range string
                range_str = ""
                if dep.access_type in (AccessType.INDEXED, AccessType.PIPELINE_INDEXED):
                    range_str = f"[-{dep.index}]"
                elif dep.access_type in (AccessType.INDEX_RANGE, AccessType.PIPELINE_INDEX_RANGE):
                    start = f"-{dep.index_start}" if dep.index_start is not None else ""
                    end = f":-{dep.index_end}" if dep.index_end is not None else ":"
                    range_str = f"[{start}{end}]"
                elif dep.access_type in (AccessType.TIME_RANGE, AccessType.PIPELINE_TIME_RANGE):
                    start = f"-{dep.time_start}s" if dep.time_start is not None else ""
                    end = f":-{dep.time_end}s" if dep.time_end is not None else ":"
                    range_str = f"[{start}{end}]"

                links.append({
                    "source": source_id,
                    "target": target_id,
                    "source_output": dep.output_name,
                    "target_input": target_input,
                    "dependency_range": range_str
                })
        
        return links

    def _build_node_label(self, layer, show_class, show_metrics) -> str:
        """Build the multi-line label for a node, mimicking the premium visualizer aesthetic."""
        # 1. Layer Name (Bold and prominent)
        parts = [f"<b>{layer.display_name}</b>"]
        
        # 2. Class Name (Subtle, uppercase-ish)
        if show_class and layer.display_name != layer.__class__.__name__:
            parts.append(f"<font size='2'><i>{layer.__class__.__name__.upper()}</i></font>")
        
        # 3. Metrics (Pink badge style)
        if show_metrics and self.pipeline.metrics:
            stats = self.pipeline.metrics.summary().get(layer.name)
            if stats:
                mean = f"{stats['mean_ms']}ms"
                p99 = f"{stats['p99_ms']}ms"
                
                # We use a styled span. Note: Mermaid support for inline style in labels 
                # depends on the renderer, but most D3-based ones (like the one in our 
                # own visualizer or VS Code) handle it well.
                badge_css = (
                    "display:inline-block; "
                    "background-color:rgba(236,72,153,0.15); "
                    "color:#db2777; "
                    "border:1px solid rgba(236,72,153,0.3); "
                    "border-radius:4px; "
                    "padding:1px 5px; "
                    "font-size:10px; "
                    "margin-top:4px;"
                )
                
                parts.append(
                    f"<span style='{badge_css}'>⏱️ Avg: {mean} <small>(p99 {p99})</small></span>"
                )
        
        return "<br/>".join(parts)


def get_mermaid(
    pipeline: "Pipeline",
    orientation: str = "LR",
    show_legend: bool = True,
    show_class_names: bool = True,
    show_metrics: bool = False,
    show_dependency_labels: bool = True,
    expand_subpipelines: bool = False,
) -> str:
    """Helper function to quickly generate Mermaid code for a pipeline."""
    return MermaidVisualizer(pipeline).generate(
        orientation=orientation,
        show_legend=show_legend,
        show_class_names=show_class_names,
        show_metrics=show_metrics,
        show_dependency_labels=show_dependency_labels,
        expand_subpipelines=expand_subpipelines,
    )


def save_mermaid_to_file(
    pipeline: "Pipeline",
    path: str,
    orientation: str = "LR",
    show_legend: bool = True,
    show_class_names: bool = True,
    show_metrics: bool = False,
    show_dependency_labels: bool = True,
    expand_subpipelines: bool = False,
) -> None:
    """Generate and save Mermaid code to a file."""
    code = get_mermaid(
        pipeline=pipeline,
        orientation=orientation,
        show_legend=show_legend,
        show_class_names=show_class_names,
        show_metrics=show_metrics,
        show_dependency_labels=show_dependency_labels,
        expand_subpipelines=expand_subpipelines,
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

