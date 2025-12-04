"""
Knowledge Graph Builder

Builds NetworkX graphs from extracted entities and relationships.
"""

import networkx as nx
from typing import List, Dict
from loguru import logger
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs from biomedical data."""
    
    def __init__(self):
        """Initialize graph builder."""
        self.graph = nx.DiGraph()
        logger.info("Knowledge graph builder initialized")
    
    def add_entities(self, entities: Dict[str, List[str]]):
        """
        Add entities to the graph as nodes.
        
        Args:
            entities: Dictionary mapping entity types to lists of entities
        """
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                if entity and entity.strip():
                    self.graph.add_node(entity.strip(), type=entity_type)
        
        logger.info(f"Added {len(self.graph.nodes)} nodes to graph")
    
    def add_relationships(self, relationships: List[Dict]):
        """
        Add relationships to the graph as edges.
        
        Args:
            relationships: List of relationship dicts with source, relation, target
        """
        for rel in relationships:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "").strip()
            
            if source and target and relation:
                self.graph.add_edge(source, target, relation=relation)
        
        logger.info(f"Added {len(self.graph.edges)} edges to graph")
    
    def build_from_articles(self, articles_with_entities: List[Dict]) -> nx.DiGraph:
        """
        Build graph from multiple articles with extracted entities/relationships.
        
        Args:
            articles_with_entities: List of dicts containing entities and relationships
        
        Returns:
            NetworkX directed graph
        """
        for article in articles_with_entities:
            entities = article.get("entities", {})
            relationships = article.get("relationships", [])
            
            self.add_entities(entities)
            self.add_relationships(relationships)
        
        logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return self.graph
    
    def get_node_info(self, node: str) -> Dict:
        """
        Get information about a specific node.
        
        Args:
            node: Node identifier
        
        Returns:
            Dictionary with node attributes and connections
        """
        if node not in self.graph:
            return {}
        
        # Get node attributes
        attrs = self.graph.nodes[node]
        
        # Get outgoing edges (what this entity affects)
        outgoing = [
            {"target": t, "relation": self.graph.edges[node, t].get("relation", "")}
            for t in self.graph.successors(node)
        ]
        
        # Get incoming edges (what affects this entity)
        incoming = [
            {"source": s, "relation": self.graph.edges[s, node].get("relation", "")}
            for s in self.graph.predecessors(node)
        ]
        
        return {
            "node": node,
            "type": attrs.get("type", ""),
            "outgoing": outgoing,
            "incoming": incoming,
        }
    
    def find_paths(self, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """
        Find all paths between two nodes.
        
        Args:
            source: Source node
            target: Target node
            max_length: Maximum path length
        
        Returns:
            List of paths (each path is a list of nodes)
        """
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_length))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def get_connected_component(self, node: str, hops: int = 2) -> nx.DiGraph:
        """
        Get subgraph of nodes within N hops of a given node.
        
        Args:
            node: Center node
            hops: Number of hops to include
        
        Returns:
            Subgraph
        """
        if node not in self.graph:
            return nx.DiGraph()
        
        # Get nodes within N hops
        nodes = {node}
        current_layer = {node}
        
        for _ in range(hops):
            next_layer = set()
            for n in current_layer:
                next_layer.update(self.graph.successors(n))
                next_layer.update(self.graph.predecessors(n))
            nodes.update(next_layer)
            current_layer = next_layer
        
        return self.graph.subgraph(nodes)
    
    def visualize(self, filename: str = "knowledge_graph.png", layout: str = "spring"):
        """
        Visualize the graph.
        
        Args:
            filename: Output filename
            layout: Layout algorithm (spring, circular, kamada_kawai)
        """
        if len(self.graph.nodes) == 0:
            logger.warning("Cannot visualize empty graph")
            return
        
        plt.figure(figsize=(16, 12))
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)
        
        # Color nodes by type
        node_colors = []
        color_map = {
            "drugs": "#FF6B6B",
            "targets": "#4ECDC4",
            "pathways": "#45B7D1",
            "outcomes": "#96CEB4",
        }
        
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "")
            node_colors.append(color_map.get(node_type, "#CCCCCC"))
        
        # Draw graph
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        nx.draw_networkx_edges(self.graph, pos, edge_color="#888888", arrows=True, 
                              arrowsize=10, alpha=0.5, width=1.5)
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, "relation")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        logger.info(f"Graph visualization saved to {filename}")
        plt.close()
    
    def get_summary(self) -> Dict:
        """
        Get summary statistics of the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "num_nodes": len(self.graph.nodes),
            "num_edges": len(self.graph.edges),
            "num_drugs": sum(1 for n in self.graph.nodes if self.graph.nodes[n].get("type") == "drugs"),
            "num_targets": sum(1 for n in self.graph.nodes if self.graph.nodes[n].get("type") == "targets"),
            "num_pathways": sum(1 for n in self.graph.nodes if self.graph.nodes[n].get("type") == "pathways"),
            "num_outcomes": sum(1 for n in self.graph.nodes if self.graph.nodes[n].get("type") == "outcomes"),
            "density": nx.density(self.graph),
        }
    
    def load_from_pickle(self, filepath: str) -> bool:
        """
        Load a graph from a pickle file.
        
        Args:
            filepath: Path to the pickle file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            graph_path = Path(filepath)
            if not graph_path.exists():
                logger.warning(f"Graph file not found: {filepath}")
                return False
            
            with open(graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            
            logger.info(f"Loaded graph from {filepath}: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            return True
        except Exception as e:
            logger.error(f"Error loading graph from {filepath}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    # Example entities
    entities = {
        "drugs": ["Pembrolizumab", "Temozolomide"],
        "targets": ["PD-1", "MGMT"],
        "pathways": ["immune checkpoint", "DNA repair"],
        "outcomes": ["increased survival", "tumor regression"],
    }
    
    # Example relationships
    relationships = [
        {"source": "Pembrolizumab", "relation": "targets", "target": "PD-1"},
        {"source": "PD-1", "relation": "regulates", "target": "immune checkpoint"},
        {"source": "immune checkpoint", "relation": "associated_with", "target": "increased survival"},
    ]
    
    builder.add_entities(entities)
    builder.add_relationships(relationships)
    
    print("Graph Summary:", builder.get_summary())
    builder.visualize("test_graph.png")

