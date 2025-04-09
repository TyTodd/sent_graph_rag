from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, TypedDict, List, Iterator, Callable
import torch
V = TypeVar('V')  # Vertex
E = TypeVar('E')  # Edge

class VertexProperties(TypedDict):
    id: str
    label: str
    terminal: bool
    ner_label: str
    aliases: List[str]

class EdgeProperties(TypedDict):
    sentence: str
    entity1: str
    entity2: str
    terminal: bool

class SentenceGraph(ABC, Generic[V, E]):
    @abstractmethod
    def add_vertex(self, properties: VertexProperties) -> V:
        """Add a vertex to the graph with the given properties."""
        pass
    
    @abstractmethod
    def add_edge(self, source: V, target: V, properties: EdgeProperties) -> E:
        """Add an edge to the graph with the given properties."""
        pass

    @abstractmethod
    def vertex_is_terminal(self, element: V) -> bool:
        """Returns True if the vertex is terminal"""
        pass
    
    @abstractmethod
    def edge_is_terminal(self, element: E) -> bool:
        """Returns True if the edge is terminal"""
        pass

    @abstractmethod
    def get_vertex_property(self, element: V , property: str) -> Any:
        """Returns the property for the vertex."""
        pass

    @abstractmethod
    def get_edge_property(self, element: E, property: str) -> Any:
        """Returns the property for the edge."""
        pass
    
    def add_vertex_embeddings(self, embeddings: torch.Tensor) -> None:
        """Adds the embeddings to verticies. Embeddings are in the same order as verticies are yielded from the iter_vertices() method."""
        pass
    
    def add_edge_embeddings(self, embeddings: torch.Tensor) -> None:
        """Adds the embeddings to the edges. Embeddings are in the same order as edges are yielded from the iter_edges() method."""
        pass
    
    @abstractmethod
    def iter_vertices(self) -> Iterator[V]:
        """Iterates over the vertices."""
        pass
    
    @abstractmethod
    def iter_edges(self) -> Iterator[E]:
        """Iterates over the edges."""
        pass
    
    @abstractmethod
    def num_vertices(self) -> int:
        """Returns the number of vertices in the graph."""
        pass
    
    @abstractmethod
    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        pass
    
    @abstractmethod
    def set_vertex_filter(self, filter: Callable[[V], bool]) -> None:
        """Sets the filter for the vertices."""
        pass
    
    @abstractmethod
    def set_edge_filter(self, filter: Callable[[E], bool], filter_unconnected_vertices: bool = False) -> None:
        """Sets the filter for the edges."""
        pass
    
    @abstractmethod
    def clear_filters(self) -> None:
        """Clears the filters"""
        pass
    
    @abstractmethod
    def iter_filtered_vertices(self) -> Iterator[V]:
        """Iterates over the vertices that match the filters."""
        pass
    
    @abstractmethod
    def iter_filtered_edges(self) -> Iterator[E]:
        """Iterates over the edges that match the filters."""
        pass
    
    
    @abstractmethod
    def get_edge_embeddings(self) -> tuple[torch.Tensor, Iterator[E]]:
        """Returns the embeddings for the edges. The first element of the tuple is a tensor of the embeddings and the second element is an iterator over the edges."""
        pass
    
    @abstractmethod
    def get_vertex_embeddings(self) -> tuple[torch.Tensor, Iterator[V]]:
        """Returns the embeddings for the vertices. The first element of the tuple is a tensor of the embeddings and the second element is an iterator over the vertices."""
        pass
    
    @abstractmethod
    def set_edge_weights(self, weights: torch.Tensor):
        """Sets the weights for the edges."""
        pass
    
    @abstractmethod
    def shortest_path(self, start_node: V, end_node: V) -> tuple[List[V], List[E]]:
        """Returns the shortest path between the start and end node."""
        pass
    
    @abstractmethod
    def edge_endpoints(self, edge: E) -> tuple[V, V]:
        """Returns the source and target of the edge."""
        pass