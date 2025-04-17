from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Dict, Any, TypedDict, List, Iterator, Callable, Union, Literal, Type, Tuple, Optional
import torch
import numpy as np
V = TypeVar('V')  # Vertex
E = TypeVar('E')  # Edge
import igraph as ig
import io
from spacy.tokens import Doc
from .graph_creation import create_graph_from_doc

try:
    import graph_tool as gt
except ImportError:
    gt = None  # So "gt.Vertex" doesn't raise NameError during runtime

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
    def __init__(self, corpus: str):
        self.id_to_vertex = None
        pass
    
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
    def set_vertex_filter(self, property: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None) -> None:
        """Sets the filter for the vertices. Filters are only guaranteed to apply to the functions iter_vertices(), 
        num_vertices(), get_vertex_embeddings(), iter_edges(), num_edges(), and get_edge_embeddings(). 
        However, GraphToolSentenceGraph will also apply the filters to all functions since it uses a GraphView .
        """
        if eq_value is not None and filter_fn is not None:
            raise ValueError("Only one of eq_value or filter_fn can be provided")
        if eq_value is None and filter_fn is None:
            raise ValueError("Either eq_value or filter_fn must be provided")
    
    @abstractmethod
    def set_edge_filter(self, property: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None, filter_unconnected_vertices: bool = False) -> None:
        """Sets the filter for the edges. Filters are only guaranteed to apply to the functions iter_edges(), 
        num_edges(), get_edge_embeddings(), iter_vertices(), num_vertices(), and get_vertex_embeddings(). 
        However, GraphToolSentenceGraph will also apply the filters to all functions since it uses a GraphView .
        """
        if eq_value is not None and filter_fn is not None:
            raise ValueError("Only one of eq_value or filter_fn can be provided")
        if eq_value is None and filter_fn is None:
            raise ValueError("Either eq_value or filter_fn must be provided")
    
    @abstractmethod
    def clear_filters(self) -> None:
        """Clears the filters"""
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
    def set_edge_weights(self, weights: List[float]):
        """Sets the weights for the edges."""
        pass
    
    @abstractmethod
    def shortest_paths(self, start_node: V, end_nodes: List[V]) -> tuple[List[V], List[E]]:
        """Returns the shortest paths between a start node and the end nodes."""
        pass
    
    @abstractmethod
    def edge_endpoints(self, edge: E) -> tuple[V, V]:
        """Returns the source and target of the edge."""
        pass
    
    @abstractmethod
    def get_neighbors(self, vertex: V) -> Iterator[V]:
        """Returns all the neighbors of the vertex."""
        pass
    
    @abstractmethod
    def to_bytes(self)-> bytes:
        """Returns the graph as a bytes object."""
        pass
    
    @classmethod
    @abstractmethod
    def from_bytes(cls, data: bytes) -> "SentenceGraph":
        """Loads the graph from a bytes object."""
        pass
    
    @abstractmethod
    def get_weight(self, edge: E) -> float:
        """Returns the weight of the edge."""
        pass
    
    def path_length(self, path: Tuple[List[V], List[E]]) -> float:
        """Returns the length of the path."""
        _ , edge_list = path
        return sum(self.get_weight(edge) for edge in edge_list)
    
    @abstractmethod
    def get_edges(self, vertex: V) -> Iterator[E]:
        """Returns the edges from the vertex."""
        pass
    
    @abstractmethod
    def is_edge(self, component: Union[V, E]) -> bool:
        """Returns True if the component is an edge."""
        pass
    
    @staticmethod
    def from_doc(doc: Doc, graph_type: Literal["igraph", "graph-tool"] = "igraph") -> "SentenceGraph":
        if graph_type == "igraph":
            return create_graph_from_doc(IGraphSentenceGraph, doc)
        elif graph_type == "graph-tool":
            return create_graph_from_doc(GraphToolSentenceGraph, doc)
        else:
            raise ValueError(f"Invalid graph type: {graph_type}")



# TODO: Warning: all functions run as if the graph is filtered in as opposed to the igraph graph where the only view you get of filters is through iter_filtered_vertices() and iter_filtered_edges()
class GraphToolSentenceGraph(SentenceGraph["gt.Vertex", "gt.Edge"]):
    def __init__(self, corpus: str):
        if gt is None:
            raise ImportError("graph-tool is required to use GraphToolSentenceGraph. Please install it first.")
        
        self.graph = gt.Graph(directed=False)
        self.graph.graph_properties["corpus"] = self.graph.new_graph_property("string")
        self.graph.graph_properties["corpus"] = corpus

        self.graph.edge_properties["sentence"] = self.graph.new_edge_property("string")  # For the sentence
        self.graph.edge_properties["entity1"] = self.graph.new_edge_property("vector<int>")  # For entity1 location
        self.graph.edge_properties["entity2"] = self.graph.new_edge_property("vector<int>")  # For entity2 location
        self.graph.edge_properties["terminal"] = self.graph.new_edge_property("bool")

        self.graph.vertex_properties["id"] = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["label"] = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["terminal"] = self.graph.new_vertex_property("bool")
        self.graph.vertex_properties["ner_label"] = self.graph.new_vertex_property("string")
        self.graph.vertex_properties["aliases"] = self.graph.new_vertex_property("object")
        
        self.edge_weights = None
        
        
        
    def add_vertex(self, properties: VertexProperties) -> "gt.Vertex":
        v = self.graph.add_vertex()
        self.graph.vertex_properties["id"][v] = properties["id"]
        self.graph.vertex_properties["label"][v] = properties["label"]
        self.graph.vertex_properties["terminal"][v] = properties["terminal"]
        self.graph.vertex_properties["ner_label"][v] = properties["ner_label"]
        self.graph.vertex_properties["aliases"][v] = properties["aliases"]
        return v
    
    def add_edge(self, source: "gt.Vertex", target: "gt.Vertex", properties: EdgeProperties) -> "gt.Edge":
        e = self.graph.add_edge(source, target)
        self.graph.edge_properties["sentence"][e] = properties["sentence"]
        self.graph.edge_properties["entity1"][e] = properties["entity1"]
        self.graph.edge_properties["entity2"][e] = properties["entity2"]
        self.graph.edge_properties["terminal"][e] = properties["terminal"]
        return e
    
    def vertex_is_terminal(self, vertex: "gt.Vertex") -> bool:
        return self.graph.vertex_properties["terminal"][vertex]
    
    def edge_is_terminal(self, edge: "gt.Edge") -> bool:
        return self.graph.edge_properties["terminal"][edge]
    
    def get_vertex_property(self, vertex: "gt.Vertex", property: str) -> Any:
        return self.graph.vertex_properties[property][vertex]
    
    def get_edge_property(self, edge: "gt.Edge", property: str) -> Any:
        return self.graph.edge_properties[property][edge]
    
    def add_vertex_embeddings(self, embeddings: torch.Tensor) -> None:
        if "embedding" not in self.graph.vertex_properties:
            self.graph.vertex_properties["embedding"] = self.graph.new_vertex_property(
                "vector<float>"
            )
        
        self.graph.vertex_properties["embedding"].set_2d_array(embeddings)
    
    def add_edge_embeddings(self, embeddings: torch.Tensor) -> None:
        if "embedding" not in self.graph.edge_properties:
            self.graph.edge_properties["embedding"] = self.graph.new_edge_property(
                "vector<float>"
            )
        self.graph.edge_properties["embedding"].set_2d_array(embeddings)
        
    def iter_vertices(self) -> Iterator["gt.Vertex"]:
        return self.graph.vertices()
    
    def iter_edges(self) -> Iterator["gt.Edge"]:
        return self.graph.edges()
    
    def num_vertices(self) -> int:
        return self.graph.num_vertices()
    
    def num_edges(self) -> int:
        return self.graph.num_edges()
    
    def set_vertex_filter(self, property: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None) -> None:
        super().set_vertex_filter(property, eq_value, filter_fn)
        if eq_value is not None:
            vprop = self.graph.new_vertex_property("bool")
            vprop.a = self.graph.vertex_properties[property].a == eq_value  # vectorized!
            self.graph.set_vertex_filter(vprop)
        else:
            prop = self.graph.vertex_properties[property]
            filter_prop = prop.transform(filter_fn, value_type="bool")
            self.graph.set_vertex_filter(filter_prop)
    
    def set_edge_filter(self, property: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None, filter_unconnected_vertices: bool = False) -> None:
        super().set_edge_filter(property, eq_value, filter_fn, filter_unconnected_vertices)
        if eq_value is not None:
            vprop = self.graph.new_edge_property("bool")
            vprop.a = self.graph.edge_properties[property].a == eq_value  # vectorized!
            self.graph.set_edge_filter(vprop)   
        else:
            prop = self.graph.edge_properties[property]
            filter_prop = prop.transform(filter_fn, value_type="bool")
            self.graph.set_edge_filter(filter_prop)
        
        if filter_unconnected_vertices:
            connected_filter = self.graph.new_vertex_property("bool")
            connected_filter.a = self.graph.get_out_degrees(self.graph.get_vertices()) > 0
            self.graph.set_vertex_filter(connected_filter) 
    
    def clear_filters(self) -> None:
        self.graph.clear_filters()
    
    def get_edge_embeddings(self) -> tuple[torch.Tensor, Iterator["gt.Edge"]]:
        return self.graph.edge_properties["embedding"].get_2d_array().T, self.graph.edges()
    
    def get_vertex_embeddings(self) -> tuple[torch.Tensor, Iterator["gt.Vertex"]]:
        return self.graph.vertex_properties["embedding"].get_2d_array().T, self.graph.vertices()
    
    def set_edge_weights(self, weights: List[float]) -> None:
        self.edge_weights = self.graph.new_edge_property("float")
        self.edge_weights.set_values(weights)
    
    def shortest_paths(self, start_node: "gt.Vertex", end_nodes: List["gt.Vertex"]) -> tuple[List["gt.Vertex"], List["gt.Edge"]]:
        paths = []
        for end_node in end_nodes:
            path, edge_list = gt.shortest_path(self.graph, start_node, end_node, weights=self.edge_weights)
            paths.append((path, edge_list))
        return paths
    
    def edge_endpoints(self, edge: "gt.Edge") -> tuple["gt.Vertex", "gt.Vertex"]:
        return edge.source(), edge.target()
    
    def to_bytes(self)-> bytes:
        buffer = io.BytesIO()
        self.graph.save(buffer)
        return buffer.getvalue()
    
    def get_neighbors(self, vertex: "gt.Vertex") -> Iterator["gt.Vertex"]:
        return vertex.all_neighbors()
    
    @classmethod
    def from_bytes(cls: Type["GraphToolSentenceGraph"], data: bytes) -> "GraphToolSentenceGraph":
        buffer = io.BytesIO(data)
        graph = cls(corpus="")
        graph.graph.load(buffer)
        return graph
    
    def get_weight(self, edge: "gt.Edge") -> float:
        return self.edge_weights[edge]
        
# IGraphVertex = Union[int, ig.Vertex]
# IGraphEdge = Union[int, ig.Edge]

IGraphVertex = ig.Vertex
IGraphEdge = ig.Edge

class IGraphSentenceGraph(SentenceGraph[IGraphVertex, IGraphEdge]):
    def __init__(self, corpus: str):
        self.graph = ig.Graph(directed=False)
        self.graph["corpus"] = corpus
        self.filtered_vertices = self.graph.vs.select(None)
        self.filtered_edges = self.graph.es.select(None)
        self.edge_weights = None
        
    def add_vertex(self, properties: VertexProperties) -> IGraphVertex:
        v = self.graph.add_vertex(name=properties["id"], label=properties["label"], terminal=properties["terminal"], ner_label=properties["ner_label"], aliases=properties["aliases"])
        return v
        
        # self.graph.add_vertices(1)
        # v = len(self.graph.vs) - 1
        # self.graph.vs[v]["id"] = properties["id"]
        # self.graph.vs[v]["label"] = properties["label"]
        # self.graph.vs[v]["terminal"] = properties["terminal"]
        # self.graph.vs[v]["ner_label"] = properties["ner_label"]
        # self.graph.vs[v]["aliases"] = properties["aliases"]
        # return v
    
    def add_edge(self, source: IGraphVertex, target: IGraphVertex, properties: EdgeProperties) -> IGraphEdge:
        edge = self.graph.add_edge(source, target)
        e = edge.index
        # self.graph.add_edges([(source, target)])
        # e = len(self.graph.es) - 1
        self.graph.es[e]["sentence"] = properties["sentence"]
        self.graph.es[e]["entity1"] = properties["entity1"]
        self.graph.es[e]["entity2"] = properties["entity2"]
        self.graph.es[e]["terminal"] = properties["terminal"]
        return e
    
    def vertex_is_terminal(self, vertex: IGraphVertex) -> bool:
        # return self.graph.vs[vertex]["terminal"]
        return vertex["terminal"]
    
    def edge_is_terminal(self, edge: IGraphEdge) -> bool:
        # return self.graph.es[edge]["terminal"]
        return edge["terminal"]
    
    def get_vertex_property(self, vertex: IGraphVertex, property_name: str) -> Any:
        # return self.graph.vs[vertex][property_name]
        vertex[property_name]
    
    def get_edge_property(self, edge: IGraphEdge, property_name: str) -> Any:
        # return self.graph.es[edge][property_name]
        edge[property_name]
    
    def add_vertex_embeddings(self, embeddings: torch.Tensor) -> None:
        self.graph.vs["embedding"] = embeddings
    
    def add_edge_embeddings(self, embeddings: torch.Tensor) -> None:
        self.graph.es["embedding"] = embeddings
    
    def iter_vertices(self) -> Iterator[IGraphVertex]:
        return self.filtered_vertices
    
    def iter_edges(self) -> Iterator[IGraphEdge]:
        return self.filtered_edges
    
    def num_vertices(self) -> int:
        return self.filtered_vertices.vcount()
    
    def num_edges(self) -> int:
        return self.filtered_edges.ecount()
    
    def set_vertex_filter(self, property_name: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None) -> None:
        super().set_vertex_filter(property_name, eq_value, filter_fn)

        transformed_fn = lambda v: filter_fn(v[property_name])
        
        if eq_value is not None:
            self.filtered_vertices = self.filtered_vertices.select(**{property_name: eq_value})
        else:
            self.filtered_vertices = self.filtered_vertices.select(transformed_fn)
            
    def set_edge_filter(self, property: str, eq_value: Any = None, filter_fn: Callable[[Any], bool] = None, filter_unconnected_vertices: bool = False) -> None:
        super().set_edge_filter(property, eq_value, filter_fn, filter_unconnected_vertices)
        
        transformed_fn = lambda e: filter_fn(e[property_name])

        if eq_value is not None:
            self.filtered_edges = self.filtered_edges.select(**{property: eq_value})
        else:
            self.filtered_edges = self.filtered_edges.select(filter_fn)
            
        if filter_unconnected_vertices:
            self.filtered_edges = self.filtered_edges.select(lambda v: v.outdegree() > 0)
        
    def clear_filters(self) -> None:
        self.filtered_vertices = self.graph.vs.select(None)
        self.filtered_edges = self.graph.es.select(None)

    def get_edge_embeddings(self) -> tuple[torch.Tensor, Iterator[IGraphEdge]]:
        # TODO: Make sure shape is correct (num_edges, embedding_dim)
        embeddings = self.filtered_edges["embedding"]
        return torch.tensor(np.array(embeddings)), self.filtered_edges
    
    def get_vertex_embeddings(self) -> tuple[torch.Tensor, Iterator[IGraphVertex]]:
        # TODO: Make sure shape is correct (num_vertices, embedding_dim)
        embeddings = self.filtered_vertices["embedding"]
        return torch.tensor(np.array(embeddings)), self.filtered_vertices
    
    def set_edge_weights(self, weights: List[float]) -> None:
        self.edge_weights = weights
        
    
    def shortest_paths(self, start_node: IGraphVertex, end_nodes: List[IGraphVertex]) -> tuple[List[IGraphVertex], List[IGraphEdge]]:
        if self.edge_weights is None:
            raise ValueError("Edge weights are not set")
        edge_lists = self.graph.get_shortest_path(start_node.index, to=[end_node.index for end_node in end_nodes], weights=self.edge_weights, output="epath")
        vertex_lists = []
        for edge_list in edge_lists:
            last_vertex = start_node
            vertex_list = [last_vertex]
            for e in edge_list:
                next_vertex = self.graph.es[e].target if self.graph.es[e].source == last_vertex else self.graph.es[e].source
                vertex_list.append(next_vertex)
                last_vertex = next_vertex
            vertex_lists.append(vertex_list)
        
        return list(zip(vertex_lists, edge_lists))
    def edge_endpoints(self, edge: IGraphEdge) -> tuple[IGraphVertex, IGraphVertex]:
        # return self.graph.es[edge].source, self.graph.es[edge].target
        return edge.source, edge.target

    def get_neighbors(self, vertex: IGraphVertex) -> Iterator[IGraphVertex]:
        return vertex.neighbors(mode="all")
    
    def to_bytes(self)-> bytes:
        buf = io.BytesIO()
        self.graph.write_graphml(buf)
        data = buf.getvalue()
        return data
    
    @classmethod
    def from_bytes(cls: Type["IGraphSentenceGraph"], data: bytes) -> "IGraphSentenceGraph":
        buf = io.BytesIO(data)
        graph = cls(corpus="")
        graph.graph = ig.Graph.Read_GraphML(buf)
        return graph
    
    def get_weight(self, edge: IGraphEdge) -> float:
        return self.edge_weights[edge.index]
    
    def get_edges(self, vertex: IGraphVertex) -> Iterator[IGraphEdge]:
        return vertex.all_edges()
    def is_edge(self, component: Union[IGraphVertex, IGraphEdge]) -> bool:
        return isinstance(component, ig.Edge)
        
            
        
    
        
        
        
        
        
    
    