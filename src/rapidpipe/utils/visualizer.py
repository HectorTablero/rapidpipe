import json
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict, Any, List

from rapidpipe.layer import AccessType, Layer


class PipelineVisualizer:
    """
    A diagnostic utility that generates a local, interactive HTML/D3.js graph
    of the Pipeline's architectural DAG, execution tiers, and live latencies.
    """

    def __init__(self, pipeline, name: str = ""):
        """
        Initialize the visualizer for a given pipeline instance.

        Args:
            pipeline: The Pipeline to visualize.
            name: Optional title string for the generated HTML page.
        """
        self.pipeline = pipeline
        self.name = name
        self.graph_data = self._analyze_pipeline()

    def _analyze_pipeline(self) -> Dict[str, Any]:
        """
        Extract layers, parsed dependencies, topological tiers, and live metrics
        into a D3-compatible JSON structure for front-end rendering.
        """
        nodes = []
        links = []

        # Maps to store connected inputs/outputs for each layer
        connected_inputs_map = {layer.name: set()
                                for layer in self.pipeline.layers}
        connected_outputs_map = {layer.name: set()
                                 for layer in self.pipeline.layers}

        # Create nodes for each layer
        for i, layer in enumerate(self.pipeline.layers):
            # Collect inputs and outputs for display
            inputs = [
                {"name": input_name} for input_name in layer.parsed_dependencies.keys()
            ]
            outputs = [{"name": output_name} for output_name in layer.outputs]

            # Execution mode
            exec_mode_str = "AUTO"
            if hasattr(layer, "execution_mode"):
                em = layer.execution_mode
                if em.name == "AUTO":
                    if hasattr(layer, "effective_execution_mode"):
                        effective = layer.effective_execution_mode
                        if effective.name == "INLINE":
                            exec_mode_str = "AUTO (INLINE)"
                        elif effective.name == "ASYNC":
                            exec_mode_str = "AUTO (ASYNC)"
                        elif effective.name == "THREAD":
                            exec_mode_str = "AUTO (THREAD)"
                    else:
                        has_custom_async = hasattr(type(layer), 'aprocess') and getattr(
                            type(layer), 'aprocess') is not getattr(Layer, 'aprocess', None)
                        exec_mode_str = "AUTO (ASYNC)" if has_custom_async else "AUTO (THREAD)"
                else:
                    exec_mode_str = em.name

            # Metrics
            metrics_str = None
            if hasattr(self.pipeline, "metrics") and self.pipeline.metrics:
                stats = self.pipeline.metrics.summary().get(layer.name)
                if stats:
                    metrics_str = f"{stats['mean_ms']}ms (p99 {stats['p99_ms']}ms)"

            nodes.append(
                {
                    "id": layer.name,
                    "display_name": layer.display_name,
                    "class_name": layer.__class__.__name__,
                    "inputs": inputs,
                    "outputs": outputs,
                    "index": i,
                    "exec_mode": exec_mode_str,
                    "metrics": metrics_str,
                }
            )

        # Helper: Find last layer (up to a given index) that outputs a value
        def find_last_processed_provider(output_name, upto_idx):
            for idx in range(upto_idx - 1, -1, -1):
                l = self.pipeline.layers[idx]
                if output_name in l.outputs:
                    return l.name
            return None

        # Helper: Find last layer (any index) that outputs a value
        def find_last_any_provider(output_name):
            for idx in range(len(self.pipeline.layers) - 1, -1, -1):
                l = self.pipeline.layers[idx]
                if output_name in l.outputs:
                    return l.name
            return None

        # Create links between layers and populate connected_inputs_map/connected_outputs_map
        link_id = 0
        for idx, layer in enumerate(self.pipeline.layers):
            for input_name, dep in layer.parsed_dependencies.items():
                source_layer_name = dep.layer_name
                dependency_range_str = ""

                # Handle pipeline dependencies
                if dep.is_pipeline_dependency:
                    # Determine provider based on access type
                    if dep.access_type in (
                        AccessType.PIPELINE_CURRENT,
                        AccessType.CURRENT,
                    ):
                        # Only consider layers already processed (before this one)
                        provider = find_last_processed_provider(
                            dep.output_name, idx)
                    else:
                        # Any layer, including those after this one
                        provider = find_last_any_provider(dep.output_name)
                    if provider:
                        source_layer_name = provider

                # Determine dependency range string
                if (
                    dep.access_type == AccessType.INDEXED
                    or dep.access_type == AccessType.PIPELINE_INDEXED
                ):
                    dependency_range_str = f"[-{dep.index}]"
                elif (
                    dep.access_type == AccessType.INDEX_RANGE
                    or dep.access_type == AccessType.PIPELINE_INDEX_RANGE
                ):
                    start = f"-{dep.index_start}" if dep.index_start is not None else ""
                    end = f":-{dep.index_end}" if dep.index_end is not None else ":"
                    dependency_range_str = f"[{start}{end}]"
                elif (
                    dep.access_type == AccessType.TIME_RANGE
                    or dep.access_type == AccessType.PIPELINE_TIME_RANGE
                ):
                    start = f"-{dep.time_start}s" if dep.time_start is not None else ""
                    end = f":-{dep.time_end}s" if dep.time_end is not None else ":"
                    dependency_range_str = f"[{start}{end}]"
                # For AccessType.CURRENT and AccessType.PIPELINE_CURRENT, dependency_range_str remains ""

                if source_layer_name:
                    links.append(
                        {
                            "id": f"link_{link_id}",
                            "source": source_layer_name,
                            "target": layer.name,
                            "source_output": dep.output_name,
                            "target_input": input_name,
                            "dependency_range": dependency_range_str,
                        }
                    )
                    link_id += 1
                    # Mark input/output as connected
                    connected_inputs_map[layer.name].add(input_name)
                    connected_outputs_map[source_layer_name].add(
                        dep.output_name)

        # Update nodes with connectivity information
        for node in nodes:
            for input_item in node["inputs"]:
                input_item["is_connected"] = (
                    input_item["name"] in connected_inputs_map[node["id"]]
                )
            for output_item in node["outputs"]:
                output_item["is_connected"] = (
                    output_item["name"] in connected_outputs_map[node["id"]]
                )

        # Calculate connectivity scores for intelligent grouping
        self._calculate_connectivity_scores(nodes, links)

        return {"nodes": nodes, "links": links}

    def _calculate_connectivity_scores(
        self, nodes: List[Dict], links: List[Dict]
    ) -> None:
        """Calculate connectivity scores for intelligent layer grouping."""
        node_map = {node["id"]: node for node in nodes}

        # Initialize connectivity scores
        for node in nodes:
            node["connectivity_score"] = 0
            node["connected_nodes"] = set()
            node["input_count"] = len(node["inputs"])
            node["output_count"] = len(node["outputs"])

        # Calculate based on connections
        for link in links:
            source_node = node_map.get(link["source"])
            target_node = node_map.get(link["target"])

            if source_node and target_node:
                source_node["connectivity_score"] += 1
                target_node["connectivity_score"] += 1
                source_node["connected_nodes"].add(link["target"])
                target_node["connected_nodes"].add(link["source"])

        # Convert sets to lists for JSON serialization
        for node in nodes:
            node["connected_nodes"] = list(node["connected_nodes"])

    def show_graph(self, free: bool = True) -> None:
        """
        Generate and launch the interactive web-based D3.js visualizer.

        Args:
            free: If True, uses the physics-based D3 force-directed layout.
                  If False, forces nodes into strict horizontal execution tiers.
        """
        html_content = self._generate_html(free=free)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            file_path = Path(f.name)
        webbrowser.open(file_path.as_uri())

    def _generate_html(self, free: bool = True) -> str:
        """Generate the complete HTML visualization using D3.js."""
        pipeline_json = json.dumps(self.graph_data, indent=2)
        free_mode_js = str(free).lower()

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.name + " - " if self.name else ""}Pipeline Visualization</title>
    <style>
        :root {{
            --page-bg: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --text-color: #e2e8f0;
            --muted-color: #94a3b8;
            --surface-color: rgba(15, 23, 42, 0.95);
            --surface-strong: rgba(15, 23, 42, 0.8);
            --card-bg: linear-gradient(145deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.95));
            --card-border: #334155;
            --card-hover-border: #60a5fa;
            --button-bg: rgba(15, 23, 42, 0.95);
            --button-hover-bg: rgba(51, 65, 85, 0.95);
            --button-border: #334155;
            --button-border-hover: #475569;
            --title-bg: rgba(15, 23, 42, 0.8);
            --link-color: #475569;
            --link-hover: #cbd5e1;
            --link-text: #cbd5e1;
            --panel-border: #334155;
            --shadow-color: rgba(0, 0, 0, 0.4);
            --loading-color: #94a3b8;
            --io-muted: #64748b;
            --unconnected-bg: rgba(100, 116, 139, 0.1);
            --unconnected-text: #94a3b8;
            --unconnected-border: #64748b;
            --text-shadow-color: #0f172a;
        }}

        body[data-theme="light"] {{
            --page-bg: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            --text-color: #0f172a;
            --muted-color: #475569;
            --surface-color: rgba(255, 255, 255, 0.92);
            --surface-strong: rgba(241, 245, 249, 0.9);
            --card-bg: linear-gradient(145deg, rgba(255, 255, 255, 0.98), rgba(241, 245, 249, 0.98));
            --card-border: #cbd5e1;
            --card-hover-border: #2563eb;
            --button-bg: rgba(255, 255, 255, 0.95);
            --button-hover-bg: rgba(226, 232, 240, 0.95);
            --button-border: #cbd5e1;
            --button-border-hover: #94a3b8;
            --title-bg: rgba(255, 255, 255, 0.85);
            --link-color: #64748b;
            --link-hover: #0f172a;
            --link-text: #334155;
            --panel-border: #cbd5e1;
            --shadow-color: rgba(15, 23, 42, 0.18);
            --loading-color: #475569;
            --io-muted: #64748b;
            --unconnected-bg: rgba(148, 163, 184, 0.12);
            --unconnected-text: #64748b;
            --unconnected-border: #94a3b8;
            --text-shadow-color: #ffffff;
        }}

        body, html {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: var(--page-bg);
            color: var(--text-color);
            overflow: hidden;
            height: 100vh;
            width: 100vw;
            margin: 0;
            padding: 0;
        }}

        #container {{
            position: relative;
            width: 100%;
            height: 100%;
        }}

        #controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 100;
            display: flex;
            gap: 12px;
            flex-direction: row;
            align-items: center;
        }}

        .control-btn {{
            padding: 12px 20px;
            background: var(--button-bg);
            border: 1px solid var(--button-border);
            border-radius: 12px;
            color: var(--text-color);
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 16px -4px var(--shadow-color);
        }}

        .control-btn:hover {{
            background: var(--button-hover-bg);
            border-color: var(--button-border-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px -4px var(--shadow-color);
        }}

        .layer-node {{
            cursor: grab;
        }}

        .layer-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            box-shadow: 0 8px 32px -8px var(--shadow-color);
            backdrop-filter: blur(12px);
            min-width: max-content;
            max-width: 280px;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .layer-card:hover {{
            border-color: var(--card-hover-border);
            box-shadow: 0 12px 40px -8px rgba(96, 165, 250, 0.3);
        }}

        .layer-header {{
            padding: 16px 20px 12px;
            border-bottom: 1px solid var(--panel-border);
            position: relative; /* For absolute badge */
        }}

        .layer-title {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 4px;
            line-height: 1.2;
        }}

        .layer-type {{
            font-size: 12px;
            color: var(--muted-color);
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }}

        .layer-meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }}

        .meta-tag {{
            font-size: 10px;
            font-weight: 600;
            padding: 3px 6px;
            border-radius: 4px;
            background: rgba(148, 163, 184, 0.15);
            color: var(--muted-color);
            border: 1px solid rgba(148, 163, 184, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .layer-mode-badge {{
            position: absolute;
            top: 12px;
            right: 12px;
            font-size: 10px;
            font-weight: 700;
            padding: 3px 6px;
            border-radius: 4px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .layer-mode-badge.mode-auto {{
            background: rgba(148, 163, 184, 0.15);
            color: var(--muted-color);
            border: 1px solid rgba(148, 163, 184, 0.3);
        }}

        .layer-mode-badge.mode-async {{
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .layer-mode-badge.mode-thread {{
            background: rgba(245, 158, 11, 0.15);
            color: #fcd34d;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .layer-mode-badge.mode-inline {{
            background: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .meta-tag.metrics-tag {{
            background: rgba(236, 72, 153, 0.15);
            color: #f9a8d4;
            border-color: rgba(236, 72, 153, 0.3);
            text-transform: none;
            letter-spacing: normal;
        }}

        .io-section {{
            padding: 16px 20px 20px;
            display: flex;
            gap: 20px;
        }}

        .io-column {{
            flex: 1;
            position: relative; /* Needed for absolute positioning of ports */
        }}

        .io-title {{
            font-size: 11px;
            font-weight: 700;
            color: var(--io-muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
        }}

        .io-item {{
            padding: 4px 8px;
            margin: 2px 0;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
            position: relative; /* For port positioning */
        }}

        .empty-io {{
            color: var(--io-muted);
            font-style: italic;
        }}

        .input-item {{
            background: rgba(59, 130, 246, 0.1);
            color: #93c5fd;
            border-left: 3px solid #3b82f6;
        }}

        .output-item {{
            background: rgba(16, 185, 129, 0.1);
            color: #6ee7b7;
            border-right: 3px solid #10b981;
        }}

        .unconnected-item {{
            background: var(--unconnected-bg); /* Gray background */
            color: var(--unconnected-text); /* Gray text */
            border-color: var(--unconnected-border); /* Gray border */
        }}

        /* New styles for connection ports */
        .port {{
            position: absolute;
            width: 10px;
            height: 10px;
            background-color: var(--link-hover); /* Default port color */
            border-radius: 50%;
            border: 1.5px solid var(--link-color);
            z-index: 1; /* Ensure ports are above the card but below links */
            transition: background-color 0.2s ease, border-color 0.2s ease;
        }}

        .input-port {{
            left: -8px; /* Position outside the card on the left */
            top: 50%;
            transform: translateY(-50%);
            background-color: #3b82f6; /* Input port color */
            border-color: #93c5fd;
        }}

        .output-port {{
            right: -8px; /* Position outside the card on the right */
            top: 50%;
            transform: translateY(-50%);
            background-color: #10b981; /* Output port color */
            border-color: #6ee7b7;
        }}

        .unconnected-port {{
            background-color: var(--unconnected-border); /* Gray port color */
            border-color: var(--unconnected-text); /* Lighter gray border */
        }}

        /* Link and Arrow Styles */
        .link {{
            fill: none;
            stroke-width: 2.5;
            stroke: var(--link-color); /* Default gray */
            opacity: 0.8;
            transition: all 0.3s ease, path 0.15s ease;
            pointer-events: stroke;
            stroke-linecap: round;
        }}

        .link-arrow {{
            fill: var(--link-color); /* Default gray */
            transition: all 0.3s ease;
        }}

        /* General Link Hover (brighter gray) */
        .link.general-hover {{
            stroke: var(--link-hover); /* Brighter gray */
            stroke-width: 3.5;
            opacity: 1;
        }}
        .link-arrow.general-hover {{
            fill: var(--link-hover); /* Brighter gray */
        }}
        .link-text.general-hover {{ /* Text hover style */
            fill: var(--link-hover);
            font-weight: 600;
            font-size: 13px;
        }}

        /* Input Highlight (soft blue) */
        .link.input-highlight {{
            stroke: #93c5fd; /* Light blue, matching input-item */
        }}
        .link-arrow.input-highlight {{
            fill: #93c5fd;
        }}
        .link-text.input-highlight {{ /* Text hover style */
            fill: #93c5fd;
            font-weight: 600;
            font-size: 12px;
        }}

        /* Output Highlight (soft green) */
        .link.output-highlight {{
            stroke: #6ee7b7; /* Light green, matching output-item */
        }}
        .link-arrow.output-highlight {{
            fill: #6ee7b7;
        }}
        .link-text.output-highlight {{ /* Text hover style */
            fill: #6ee7b7;
            font-weight: 600;
            font-size: 12px;
        }}

        /* Self-referencing Link (soft pink) */
        .link.self-referencing.self-referencing-highlight {{
            stroke: #ff94e9; /* Soft pink */
        }}
        .link-arrow.self-referencing.self-referencing-highlight {{
            fill: #ff94e9;
        }}
        .link-text.self-referencing-highlight {{ /* Text hover style */
            fill: #ff94e9;
            font-weight: 600;
            font-size: 12px;
        }}

        /* Styles for dependency range text */
        .link-text {{
            font-size: 10px;
            fill: var(--link-text);
            text-anchor: middle;
            dominant-baseline: central;
            pointer-events: all; /* Make text clickable/hoverable */
            opacity: 0.9;
            transition: all 0.3s ease, font-weight 0.1s ease; /* Add transition for smoothness */
            cursor: default;
            text-shadow: 
                -1px -1px 0 var(--text-shadow-color),  
                1px -1px 0 var(--text-shadow-color),
                -1px 1px 0 var(--text-shadow-color),
                1px 1px 0 var(--text-shadow-color);
        }}


        #info {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            padding: 12px 16px;
            background: var(--surface-color);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            backdrop-filter: blur(12px);
            font-size: 13px;
            color: var(--text-color);
            z-index: 100;
        }}

        .loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: var(--loading-color);
            font-size: 18px;
            z-index: 1000;
        }}

        foreignObject {{
            overflow: visible;
        }}

        #pipeline-title {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: var(--text-color);
            font-size: 16px;
            font-weight: 700;
            z-index: 101; /* Above controls */
            text-shadow: 0 2px 8px rgba(0,0,0,0.5);
            padding: 5px 10px;
            background: var(--title-bg);
            border-radius: 12px;
            backdrop-filter: blur(8px);
            border: 1px solid var(--panel-border);
        }}
    </style>
</head>
<body>
    {'<div id="pipeline-title">' + self.name + "</div>" if self.name else ""}
    <div id="container">
        <div class="loading" id="loading">Initializing visualization...</div>
        <svg id="graph"></svg>
        <div id="controls">
            <button class="control-btn" onclick="centerView()">🎯 Center</button>
            <button class="control-btn" id="theme-toggle" onclick="toggleTheme()">Theme</button>
        </div>
        <div id="info">
            Drag to pan • Scroll to zoom • Drag nodes to reposition
        </div>
    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const data = {pipeline_json};
        const freeMode = {free_mode_js};
        
        // Configuration
        const config = {{
            width: window.innerWidth,
            height: window.innerHeight,
            nodeWidth: 260,
            linkDistance: 300,
            chargeStrength: -800,
            collisionRadius: 150
        }};

        // Dynamically adjust SVG dimensions based on number of nodes
        const nodesPerRow = 4; // Target number of nodes per row for calculation
        const nodeSpacingX = config.nodeWidth + config.linkDistance / 2;
        const nodeSpacingY = 400; // Average height of node + spacing

        // Calculate initial SVG dimensions
        let calculatedSvgWidth = Math.max(window.innerWidth, nodesPerRow * nodeSpacingX);
        let calculatedSvgHeight = Math.max(window.innerHeight, Math.ceil(data.nodes.length / nodesPerRow) * nodeSpacingY);


        // Calculate dynamic heights for nodes - fixed bottom padding
        data.nodes.forEach(node => {{
            const maxPorts = Math.max(node.inputs.length || 1, node.outputs.length || 1);
            // Increased bottom padding to prevent cutoff
            node.cardHeight = 60 + 30 + (maxPorts * 28) + 30; // header + top padding + ports + bottom padding
        }});

        // Create SVG
        const svg = d3.select("#graph")
            .attr("width", calculatedSvgWidth)
            .attr("height", calculatedSvgHeight);

        // Create container group for zoom/pan
        const container = svg.append("g");

        // Define arrow markers
        const defs = svg.append("defs");
        defs.append("marker")
            .attr("id", "arrow")
            .attr("viewBox", "0 -5 10 10")
            .attr("refX", 5) // Adjusted refX to point better at the end of the path
            .attr("refY", 0)
            .attr("markerWidth", 6)
            .attr("markerHeight", 6)
            .attr("orient", "auto")
            .attr("markerUnits", "strokeWidth")
            .append("path")
            .attr("d", "M0,-5L10,0L0,5")
            .attr("class", "link-arrow");

        // Create zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }});

        svg.call(zoom);

        const themeStorageKey = "pipeline-visualizer-theme";
        const themeToggleButton = document.getElementById("theme-toggle");

        function applyTheme(theme) {{
            document.body.dataset.theme = theme;
            if (themeToggleButton) {{
                themeToggleButton.textContent = theme === "dark" ? "Light Theme" : "Dark Theme";
            }}
            try {{
                localStorage.setItem(themeStorageKey, theme);
            }} catch (error) {{
                // Ignore storage issues in restricted contexts.
            }}
        }}

        let initialTheme = "dark";
        try {{
            initialTheme = localStorage.getItem(themeStorageKey) ||
                (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
        }} catch (error) {{
            initialTheme = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
        }}

        window.toggleTheme = () => {{
            const nextTheme = document.body.dataset.theme === "light" ? "dark" : "light";
            applyTheme(nextTheme);
        }};

        applyTheme(initialTheme);

        // Calculate max inputs/outputs for positioning
        const maxInputs = d3.max(data.nodes, d => d.inputs.length) || 1;
        const maxOutputs = d3.max(data.nodes, d => d.outputs.length) || 1;

        // Initial x-force strength
        let xForceStrength = 0.5;
        let tickCount = 0; // To track simulation ticks

        const updateNodePositions = () => {{
            nodes.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }};

        const renderGraph = () => {{
            updateLinkPositions();
            updateNodePositions();
        }};

        let simulation = null;
        let physicsEnabled = true;

        // Create force simulation with dynamic collision
        simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links)
                .id(d => d.id)
                .distance(config.linkDistance)
                .strength(0.8))
            .force("charge", d3.forceManyBody()
                .strength(config.chargeStrength))
            .force("center", d3.forceCenter(calculatedSvgWidth / 2, calculatedSvgHeight / 2))
            .force("collision", d3.forceCollide()
                .radius(d => Math.max(d.cardHeight / 2 + 80, config.collisionRadius)) // Keep this buffer for node collision
                .strength(1))
            .force("x", d3.forceX(d => {{
                const outputDiff = (d.output_count - d.input_count) * (calculatedSvgWidth * 0.2);
                return calculatedSvgWidth / 2 - outputDiff;
            }}).strength(xForceStrength)) // Initial strength for positioning
            .force("y", d3.forceY(calculatedSvgHeight / 2).strength(0.1))
            .force("boundary", () => {{
                // Keep nodes within viewport boundaries
                data.nodes.forEach(node => {{
                    const margin = Math.max(node.cardHeight / 2 + 40, config.collisionRadius);
                    node.x = Math.max(margin, Math.min(calculatedSvgWidth - margin, node.x));
                    node.y = Math.max(margin, Math.min(calculatedSvgHeight - margin, node.y));
                }});
            }});

        if (freeMode) {{
            setTimeout(() => {{
                if (!physicsEnabled) return;
                physicsEnabled = false;
                data.nodes.forEach(node => {{
                    node.fx = null;
                    node.fy = null;
                }});
                simulation.stop();
                simulation.alphaTarget(0);
                renderGraph();
            }}, 3000);
        }}

        // Create nodes first (so they appear behind links)
        const nodes = container.selectAll(".layer-node")
            .data(data.nodes)
            .enter().append("g")
            .attr("class", "layer-node")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));

        // Create links after nodes (so they appear on top)
        const links = container.selectAll(".link")
            .data(data.links)
            .enter().append("path")
            .attr("class", "link")
            .each(function(d) {{ // Add .each to create unique markers for each link
                defs.append("marker")
                    .attr("id", `arrow-${{d.id}}`) // Unique ID for each marker
                    .attr("viewBox", "0 -5 10 10")
                    .attr("refX", 6)
                    .attr("refY", 0)
                    .attr("markerWidth", 6)
                    .attr("markerHeight", 6)
                    .attr("orient", "auto")
                    .attr("markerUnits", "strokeWidth")
                    .append("path")
                    .attr("d", "M0,-5L10,0L0,5")
                    .attr("class", "link-arrow"); // Keep this class for general styling

                // Apply self-referencing class if source and target are the same
                if (d.source.id === d.target.id) {{
                    d3.select(this).classed("self-referencing", true);
                    d3.select(`#arrow-${{d.id}} .link-arrow`).classed("self-referencing", true);
                }}
            }})
            .attr("marker-end", d => `url(#arrow-${{d.id}})`); // Reference unique marker

        // Function to apply/remove link and text highlight classes
        function setLinkHighlight(d, isHovered, isSourceHovered, isTargetHovered) {{
            const linkPath = links.filter(link => link.id === d.id);
            const arrowPath = defs.select(`#arrow-${{d.id}} .link-arrow`);
            const textPath = linkTexts.filter(text => text.id === d.id);

            // Reset all specific highlights first
            linkPath.classed("output-highlight", false)
                    .classed("input-highlight", false)
                    .classed("self-referencing-highlight", false)
                    .classed("general-hover", false);
            arrowPath.classed("output-highlight", false)
                     .classed("input-highlight", false)
                     .classed("self-referencing-highlight", false)
                     .classed("general-hover", false);
            textPath.classed("output-highlight", false)
                    .classed("input-highlight", false)
                    .classed("self-referencing-highlight", false)
                    .classed("general-hover", false);

            if (isHovered) {{ // Link itself is hovered
                linkPath.classed("general-hover", true);
                arrowPath.classed("general-hover", true);
                textPath.classed("general-hover", true);
            }} else if (isSourceHovered && d.source.id === d.target.id) {{ // Self-referencing node hovered
                linkPath.classed("self-referencing-highlight", true);
                arrowPath.classed("self-referencing-highlight", true);
                textPath.classed("self-referencing-highlight", true);
            }} else if (isSourceHovered) {{ // Source node is hovered
                linkPath.classed("output-highlight", true);
                arrowPath.classed("output-highlight", true);
                textPath.classed("output-highlight", true);
            }} else if (isTargetHovered) {{ // Target node is hovered
                linkPath.classed("input-highlight", true);
                arrowPath.classed("input-highlight", true);
                textPath.classed("input-highlight", true);
            }}
        }}


        // Add general hover effects for links (brighter gray)
        links.on("mouseover", function(event, d) {{
            setLinkHighlight(d, true, false, false); // Link is hovered directly
        }})
        .on("mouseout", function(event, d) {{
            setLinkHighlight(d, false, false, false); // Remove highlight
        }});

        // Add text labels for dependency ranges
        const linkTexts = container.selectAll(".link-text")
            .data(data.links.filter(d => d.dependency_range !== "")) // Only add text for links with a range
            .enter().append("text")
            .attr("class", "link-text")
            .attr("id", d => `link-text-${{d.id}}`) // Add ID for easy selection
            .text(d => d.dependency_range)
            // Add hover effects for text
            .on("mouseover", function(event, d) {{
                setLinkHighlight(d, true, false, false); // Treat text hover like link hover
            }})
            .on("mouseout", function(event, d) {{
                setLinkHighlight(d, false, false, false); // Remove highlight
            }});


        // Add node content
        nodes.each(function(d) {{
            const node = d3.select(this);
            
            // Create foreign object with dynamic height and extra padding
            const fo = node.append("foreignObject")
                .attr("width", config.nodeWidth)
                .attr("height", d.cardHeight)
                .attr("x", -config.nodeWidth / 2)
                .attr("y", -d.cardHeight / 2);

            // Build HTML content
            const inputsHtml = d.inputs.length > 0 
                ? d.inputs.map((i, idx) => `<div class="io-item input-item ${{i.is_connected ? '' : 'unconnected-item'}}" data-port="${{i.name}}" data-index="${{idx}}">${{i.name}}<div class="port input-port ${{i.is_connected ? '' : 'unconnected-port'}}"></div></div>`).join('')
                : '<div class="io-item empty-io">None</div>';
                
            const outputsHtml = d.outputs.length > 0
                ? d.outputs.map((o, idx) => `<div class="io-item output-item ${{o.is_connected ? '' : 'unconnected-item'}}" data-port="${{o.name}}" data-index="${{idx}}">${{o.name}}<div class="port output-port ${{o.is_connected ? '' : 'unconnected-port'}}"></div></div>`).join('')
                : '<div class="io-item empty-io">None</div>';

            fo.append("xhtml:div")
                .style("width", "100%")
                .style("height", "100%")
                .html(`
                    <div class="layer-card">
                        <div class="layer-header">
                            <div class="layer-mode-badge mode-${{d.exec_mode.includes('INLINE') ? 'inline' : d.exec_mode.includes('THREAD') ? 'thread' : d.exec_mode.includes('ASYNC') ? 'async' : d.exec_mode.split(' ')[0].toLowerCase()}}">${{d.exec_mode}}</div>
                            <div class="layer-title">${{d.display_name}}</div>
                            <div class="layer-type">${{d.class_name}}</div>
                            <div class="layer-meta">
                                ${{d.metrics ? `<span class="meta-tag metrics-tag">⏱ ${{d.metrics}}</span>` : ''}}
                            </div>
                        </div>
                        <div class="io-section">
                            <div class="io-column">
                                <div class="io-title">Inputs</div>
                                ${{inputsHtml}}
                            </div>
                            <div class="io-column">
                                <div class="io-title">Outputs</div>
                                ${{outputsHtml}}
                            </div>
                        </div>
                    </div>
                `);
        }})
        // Add mouseover and mouseout events to the nodes for highlighting connected links
        .on("mouseover", function(event, hoveredNode) {{
            links.each(function(d) {{
                const isSourceHovered = (d.source.id === hoveredNode.id);
                const isTargetHovered = (d.target.id === hoveredNode.id);
                setLinkHighlight(d, false, isSourceHovered, isTargetHovered);
            }});
        }})
        .on("mouseout", function(event, hoveredNode) {{
            links.each(function(d) {{
                setLinkHighlight(d, false, false, false); // Remove all highlight
            }});
        }});

        // Helper to get a point on a cubic Bezier curve
        function getCubicBezierPoint(p0, p1, p2, p3, t) {{
            const mt = 1 - t;
            const mt2 = mt * mt;
            const t2 = t * t;
            const x = mt2 * mt * p0.x + 3 * mt2 * t * p1.x + 3 * mt * t2 * p2.x + t2 * t * p3.x;
            const y = mt2 * mt * p0.y + 3 * mt2 * t * p1.y + 3 * mt * t2 * p2.y + t2 * t * p3.y;
            return {{ x, y }};
        }}

        // Function to get port positions
        function getPortPosition(node, portName, isInput) {{
            const cardElement = nodes.filter(d => d.id === node.id).select(".layer-card").node();
            if (!cardElement) return {{ x: node.x, y: node.y }}; // Fallback

            const ioItemSelector = isInput ? `.input-item[data-port="${{portName}}"]` : `.output-item[data-port="${{portName}}"]`;
            const portElement = d3.select(cardElement).select(ioItemSelector).select(".port").node();

            if (portElement) {{
                const cardRect = cardElement.getBoundingClientRect();
                const portRect = portElement.getBoundingClientRect();

                // Calculate position relative to the SVG container
                // node.x, node.y are the center of the node group
                // We need to translate portRect coordinates to be relative to the node's center
                const svgRect = svg.node().getBoundingClientRect();

                const portXInSvg = portRect.left + portRect.width / 2 - svgRect.left;
                const portYInSvg = portRect.top + portRect.height / 2 - svgRect.top;

                // Adjust for the current transform of the container group
                const transform = d3.zoomTransform(svg.node());
                const transformedX = (portXInSvg - transform.x) / transform.k;
                const transformedY = (portYInSvg - transform.y) / transform.k;

                return {{ x: transformedX, y: transformedY }};
            }}

            // Fallback if port element not found
            const cardWidth = config.nodeWidth;
            const cardHeight = node.cardHeight;
            const headerHeight = 60;
            const ioTitleHeight = 20;
            const portHeight = 28;

            const ports = isInput ? node.inputs : node.outputs;
            const portIndex = ports.findIndex(p => p.name === portName);
            
            let portY;
            if (portIndex === -1) {{
                portY = headerHeight + ioTitleHeight + 8 + (ports.length * portHeight / 2);
            }} else {{
                portY = headerHeight + ioTitleHeight + 8 + (portIndex * portHeight) + (portHeight / 2);
            }}

            const portXOffset = isInput ? -cardWidth / 2 - 8 : cardWidth / 2 + 8; // Position for the port circle
            return {{
                x: node.x + portXOffset,
                y: node.y - cardHeight / 2 + portY
            }};
        }}

        // Function to check if a point is inside a node's buffered area
        function isPointInNode(x, y, node) {{
            const halfWidth = config.nodeWidth / 2;
            const halfHeight = node.cardHeight / 2;
            const buffer = 70; // Increased buffer for better path avoidance
            return x >= node.x - halfWidth - buffer && x <= node.x + halfWidth + buffer &&
                y >= node.y - halfHeight - buffer && y <= node.y + halfHeight + buffer;
        }}

        // Function to create collision-avoiding path with smooth Bezier curves
        function createSmartPath(sourcePos, targetPos, sourceNode, targetNode) {{
            const sx = sourcePos.x;
            const sy = sourcePos.y;
            const tx = targetPos.x - 12.5;
            const ty = targetPos.y;

            let controlOffset = 100;
            let path = `M${{sx}},${{sy}}C`;

            // Case 1: Target is to the right of source (most common, generally flows left to right)
            if (tx > sx) {{
                const dx = Math.abs(tx - sx);
                const dy = Math.abs(ty - sy);
                
                // Adjust control points based on distance and vertical alignment
                const cp1x = sx + dx * 0.4;
                const cp2x = tx - dx * 0.4;

                const cp1y = sy;
                const cp2y = ty;
                
                path += `${{cp1x}},${{cp1y}} ${{cp2x}},${{cp2y}} ${{tx}},${{ty}}`;
            }}
            // Case 2: Target is to the left of source (feedback loop or unusual layout)
            else {{
                const verticalOffset = Math.max(100, Math.abs(sy - ty) / 2 + 50);
                const horizontalOffset = Math.max(150, Math.abs(sx - tx) / 2 + 50);

                const cp1x = sx + horizontalOffset;
                const cp1y = sy + verticalOffset;

                const cp2x = tx - horizontalOffset;
                const cp2y = ty + verticalOffset;
                
                path += `${{cp1x}},${{cp1y}} ${{cp2x}},${{cp2y}} ${{tx}},${{ty}}`;
            }}
            path += `L${{tx + 7.5}},${{ty}}`;

            return path;
        }}

        // Function to update link positions
        function updateLinkPositions() {{
            links.attr("d", d => {{
                const sourcePos = getPortPosition(d.source, d.source_output, false);
                const targetPos = getPortPosition(d.target, d.target_input, true);
                // Store path data for text positioning
                d.pathData = createSmartPath(sourcePos, targetPos, d.source, d.target);
                return d.pathData;
            }});

            // Update text positions
            linkTexts.attr("transform", function(d) {{
                if (!d.pathData) return ""; // Ensure path data exists

                // Get the total length of the path
                const pathNode = links.filter(link => link.id === d.id).node();
                if (!pathNode) return "";

                const pathLength = pathNode.getTotalLength();
                // Get the point at 50% of the path length
                const p = pathNode.getPointAtLength(pathLength / 2);

                // Calculate the angle based on the tangent at the midpoint
                const p1 = pathNode.getPointAtLength(pathLength / 2 - 1); // Point slightly before midpoint
                const p2 = pathNode.getPointAtLength(pathLength / 2 + 1); // Point slightly after midpoint
                let angle = Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;

                // Ensure text is always right-side up
                if (angle > 90 || angle < -90) {{
                    angle += 180;
                }}
                
                return `translate(${{p.x}},${{p.y}}) rotate(${{angle}})`;
            }})
            .attr("dx", 0) // Reset dx, dy as rotation handles position
            .attr("dy", -8); // Slightly above the line
        }}

        // Update positions on simulation tick
        if (simulation) {{
            simulation.on("tick", () => {{
                // Lower x-force strength after initial ticks
                tickCount++;
                if (tickCount === 200) {{ // Adjust this number as needed
                    simulation.force("x").strength(0.0);
                    simulation.alpha(0.1).restart(); // Restart with lower alpha to settle
                }}

                // Update link positions with collision avoidance
                renderGraph();
            }});
        }}

        // Drag functions
        function dragstarted(event, d) {{
            if (!physicsEnabled) {{
                return;
            }}

            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            if (!physicsEnabled) {{
                d.x = event.x;
                d.y = event.y;
                renderGraph();
                return;
            }}

            d.fx = event.x;
            d.fy = event.y;
            // Manually update link positions for immediate feedback during drag
            updateLinkPositions(); 
        }}

        function dragended(event, d) {{
            if (!physicsEnabled) {{
                d.fx = null;
                d.fy = null;
                renderGraph();
                return;
            }}

            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
            // Ensure links are updated one last time after drag ends
            updateLinkPositions(); 
        }}

        window.centerView = () => {{
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            const padding = 100; // Padding around the content

            // Calculate bounding box from node positions
            data.nodes.forEach(node => {{
                const x1 = node.x - config.nodeWidth / 2;
                const y1 = node.y - node.cardHeight / 2;
                const x2 = node.x + config.nodeWidth / 2;
                const y2 = node.y + node.cardHeight / 2;

                minX = Math.min(minX, x1);
                minY = Math.min(minY, y1);
                maxX = Math.max(maxX, x2);
                maxY = Math.max(maxY, y2);
            }});

            if (minX === Infinity) return; // No nodes found

            const contentWidth = maxX - minX;
            const contentHeight = maxY - minY;
            const midX = minX + contentWidth / 2;
            const midY = minY + contentHeight / 2;
            
            const svgWidth = parseFloat(svg.attr("width"));
            const svgHeight = parseFloat(svg.attr("height"));

            const scaleX = (svgWidth - 2 * padding) / contentWidth;
            const scaleY = (svgHeight - 2 * padding) / contentHeight;
            const scale = Math.min(scaleX, scaleY);

            const translate = [
                svgWidth / 2 - scale * midX,
                svgHeight / 2 - scale * midY - 50
            ];
            
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale));
        }};

        // Handle window resize
        window.addEventListener('resize', () => {{
            // Update SVG dimensions
            calculatedSvgWidth = window.innerWidth;
            calculatedSvgHeight = window.innerHeight;

            svg.attr("width", calculatedSvgWidth).attr("height", calculatedSvgHeight);

            if (!physicsEnabled) {{
                renderGraph();
                setTimeout(() => {{
                    centerView();
                }}, 500);
                return;
            }}
            
            // Update force center and restart simulation
            simulation.force("center", d3.forceCenter(calculatedSvgWidth / 2, calculatedSvgHeight / 2));
            simulation.force("x", d3.forceX(d => {{
                const outputBias = (d.output_count / maxOutputs) * (calculatedSvgWidth * 0.2);
                const inputBias = (d.input_count / maxInputs) * (calculatedSvgWidth * 0.2);
                return calculatedSvgWidth / 2 - outputBias + inputBias;
            }}).strength(0.15)); // Keep lower strength after initial layout
            
            // Restart simulation and then re-center view
            simulation.alpha(0.3).restart();
            setTimeout(() => {{
                centerView();
            }}, 500); // Give simulation a moment to adjust before centering
        }});

        // Hide loading message when simulation starts
        if (simulation) {{
            simulation.on("tick.loading", function() {{
                d3.select("#loading").style("display", "none");
                simulation.on("tick.loading", null);
            }});
        }}

        // Initial center after a short delay
        setTimeout(() => {{
            centerView();
        }}, freeMode ? 3500 : 1000);
    </script>
</body>
</html>
        """


def visualize_pipeline(pipeline, name: str = "", free: bool = True) -> None:
    """
    Create and display a modern visualization of the pipeline.

    Args:
        pipeline: The Pipeline instance to visualize
    """
    visualizer = PipelineVisualizer(pipeline, name=name)
    visualizer.show_graph(free=free)
