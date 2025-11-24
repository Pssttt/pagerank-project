import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple


def pagerank(
    graph: Dict[str, List[str]],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    # Get all nodes
    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)

    nodes = sorted(all_nodes)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize PageRank values uniformly
    pagerank_values = np.ones(n) / n
    history = [{node: pagerank_values[node_to_idx[node]] for node in nodes}]

    # Build adjacency matrix
    M = np.zeros((n, n))
    for source, targets in graph.items():
        if targets:
            source_idx = node_to_idx[source]
            for target in targets:
                target_idx = node_to_idx[target]
                M[target_idx][source_idx] = 1.0 / len(targets)
        else:  # Dangling node
            source_idx = node_to_idx[source]
            M[:, source_idx] = 1.0 / n

    # Iterative calculation
    for iteration in range(max_iterations):
        new_pagerank = (1 - damping) / n + damping * M.dot(pagerank_values)
        history.append({node: new_pagerank[node_to_idx[node]] for node in nodes})
        if np.abs(new_pagerank - pagerank_values).sum() < tolerance:
            pagerank_values = new_pagerank
            break
        pagerank_values = new_pagerank

    # Normalize to sum to 1
    pagerank_values = pagerank_values / pagerank_values.sum()
    final_ranks = {node: pagerank_values[node_to_idx[node]] for node in nodes}
    return final_ranks, history


def visualize_graph(
    graph: Dict[str, List[str]],
    ranks: Dict[str, float],
    title: str = "PageRank Visualization",
    filename: str = None,
):
    G = nx.DiGraph()
    for source, targets in graph.items():
        for target in targets:
            G.add_edge(source, target)

    all_nodes = set(graph.keys())
    for targets in graph.values():
        all_nodes.update(targets)
    for node in all_nodes:
        if node not in G:
            G.add_node(node)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    node_sizes = [ranks.get(node, 0) * 10000 for node in G.nodes()]
    node_colors = [ranks.get(node, 0) for node in G.nodes()]

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        alpha=0.6,
        connectionstyle="arc3,rad=0.1",
        width=2,
    )

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.YlOrRd,
        alpha=0.9,
        vmin=0,
        vmax=max(ranks.values()),
    )
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    plt.colorbar(nodes, label="PageRank Score", shrink=0.8)
    text_pos = {k: (v[0], v[1] - 0.15) for k, v in pos.items()}
    labels = {node: f"{ranks.get(node, 0):.4f}" for node in G.nodes()}
    nx.draw_networkx_labels(
        G, text_pos, labels, font_size=9, font_color="darkblue", font_weight="normal"
    )

    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def plot_convergence(
    history: List[Dict[str, float]],
    title: str = "PageRank Convergence",
    filename: str = None,
):
    plt.figure(figsize=(12, 6))
    nodes = sorted(history[0].keys())
    for node in nodes:
        values = [iteration[node] for iteration in history]
        plt.plot(range(len(values)), values, marker="o", label=node, linewidth=2)

    plt.xlabel("Iteration", fontsize=12, fontweight="bold")
    plt.ylabel("PageRank Score", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


def create_comparison_chart(
    examples: List[Tuple[str, Dict[str, float]]], filename: str = None
):
    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 5))
    if len(examples) == 1:
        axes = [axes]

    for idx, (name, ranks) in enumerate(examples):
        nodes = sorted(ranks.keys())
        values = [ranks[node] for node in nodes]
        bars = axes[idx].bar(
            nodes, values, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(nodes)))
        )
        axes[idx].set_xlabel("Node", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel("PageRank Score", fontsize=11, fontweight="bold")
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].grid(axis="y", alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()


# Example 1: Classic PageRank example
graph1 = {"A": ["B", "C", "D"], "B": ["D"], "C": ["A"], "D": ["A", "C"]}

ranks1, history1 = pagerank(graph1)
visualize_graph(graph1, ranks1, "Example 1: Complex Graph", "pagerank_complex.png")
plot_convergence(history1, "Example 1: Convergence", "convergence_complex.png")


# Example 2: Two disconnected components demonstrating damping factor
graph2 = {
    # Main component (4 nodes in a cycle with some extra connections)
    "A": ["B", "C"],
    "B": ["C", "D"],
    "C": ["D"],
    "D": ["A"],
    # Isolated component (2 nodes forming their own cycle)
    "I1": ["I2"],
    "I2": ["I1"],
}

ranks2, history2 = pagerank(graph2)
visualize_graph(
    graph2,
    ranks2,
    "Example 2: Disconnected Components (Damping Factor)",
    "pagerank_disconnected.png",
)
plot_convergence(
    history2,
    "Example 2: Convergence with Disconnected Components",
    "convergence_disconnected.png",
)


# Create comparison chart
examples = [("Complex", ranks1), ("Disconnected", ranks2)]
create_comparison_chart(examples, "pagerank_comparison.png")
