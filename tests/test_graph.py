import pytest
import torch
from sent_graph_rag import IGraphSentenceGraph, VertexProperties, EdgeProperties
from typing import List

@pytest.fixture
def graph():
    return IGraphSentenceGraph("test corpus")

def test_add_vertex(graph):
    vertex_props: VertexProperties = {
        "id": "1",
        "label": "test entity",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["test", "entity"]
    }
    vertex = graph.add_vertex(vertex_props)
    assert graph.get_vertex_property(vertex, "label") == "test entity"
    assert graph.vertex_is_terminal(vertex)
    assert graph.get_vertex_property(vertex, "ner_label") == "PERSON"
    assert not graph.is_edge(vertex)

def test_add_edge(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "John",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["John Doe"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Company",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": ["The Company"]
    })
    edge_props: EdgeProperties = {
        "sentence": "John works at Company",
        "entity1": "John",
        "entity2": "Company",
        "terminal": False
    }
    edge = graph.add_edge(v1, v2, edge_props)
    assert graph.get_edge_property(edge, "sentence") == "John works at Company"
    assert not graph.edge_is_terminal(edge)
    assert graph.is_edge(edge)

def test_vertex_embeddings(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "Entity1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": []
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Entity2",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": []
    })
    embeddings = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    graph.add_vertex_embeddings(embeddings)
    emb, vertices = graph.get_vertex_embeddings()
    assert torch.equal(emb, embeddings)
    assert list(vertices) == [v1, v2]

def test_edge_embeddings(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "Entity1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": []
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Entity2",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": []
    })
    edge = graph.add_edge(v1, v2, {
        "sentence": "Entity1 works at Entity2",
        "entity1": "Entity1",
        "entity2": "Entity2",
        "terminal": False
    })
    embeddings = torch.tensor([[1.0, 2.0]])
    graph.add_edge_embeddings(embeddings)
    emb, edges = graph.get_edge_embeddings()
    assert torch.equal(emb, embeddings)
    assert list(edges) == [edge]

def test_vertex_filter(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "Person1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["A", "B"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Company1",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": ["C", "D"]
    })
    v3 = graph.add_vertex({
        "id": "3",
        "label": "Company2",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": ["A", "E"]
    })
    graph.set_vertex_filter("ner_label", eq_value="PERSON")
    assert graph.num_vertices() == 1
    vertices = list(graph.iter_vertices())
    assert len(vertices) == 1
    assert vertices[0] == v1
    graph.clear_filters()
    graph.set_vertex_filter("aliases", filter_fn=lambda a: "A" in a)
    assert graph.num_vertices() == 2
    vertices = list(graph.iter_vertices())
    assert len(vertices) == 2
    assert v1 in vertices
    assert v3 in vertices
    graph.clear_filters()

def test_edge_filter(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "Person1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": []
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Company1",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": []
    })
    v3 = graph.add_vertex({
        "id": "3",
        "label": "Company2",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": []
    })
    e1 = graph.add_edge(v1, v2, {
        "sentence": "Person1 works at Nonprofit1",
        "entity1": "Person1",
        "entity2": "Nonprofit1",
        "terminal": True
    })
    e2 = graph.add_edge(v2, v1, {
        "sentence": "Company1 employs Person1",
        "entity1": "Company1",
        "entity2": "Person1",
        "terminal": False
    })
    e3 = graph.add_edge(v2, v3, {
        "sentence": "Company1 is located in Company2",
        "entity1": "Company1",
        "entity2": "Company2",
        "terminal": False
    })
    graph.set_edge_filter("terminal", eq_value=True)
    assert graph.num_edges() == 1
    edges = list(graph.iter_edges())
    assert len(edges) == 1
    assert edges[0] == e1
    graph.clear_filters()
    graph.set_edge_filter("sentence", filter_fn=lambda s: "Comp" in s)
    assert graph.num_edges() == 2
    edges = list(graph.iter_edges())
    assert len(edges) == 2
    assert e1 in edges
    assert e3 in edges
    graph.clear_filters()

def test_shortest_paths(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "Person1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": []
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "Company1",
        "terminal": True,
        "ner_label": "ORG",
        "aliases": []
    })
    v3 = graph.add_vertex({
        "id": "3",
        "label": "Location1",
        "terminal": True,
        "ner_label": "LOC",
        "aliases": []
    })
    e1 = graph.add_edge(v1, v2, { #1.0
        "sentence": "Person1 works at Company1",
        "entity1": "Person1",
        "entity2": "Company1",
        "terminal": False
    })
    e2 = graph.add_edge(v2, v3, { #1.0
        "sentence": "Company1 is located in Location1",
        "entity1": "Company1",
        "entity2": "Location1",
        "terminal": False
    })
    
    e3 = graph.add_edge(v1, v3, { #3.0
        "sentence": "Person1 is located in Location1",
        "entity1": "Person1",
        "entity2": "Location1",
        "terminal": False
    })
    
    e4 = graph.add_edge(v1, v2, { #4.0
        "sentence": "Company1 is located in Location1",
        "entity1": "Company1",
        "entity2": "Location1",
        "terminal": False
    })
    
    graph.set_edge_weights([1.0, 1.0, 3.0, 4.0])
    paths = graph.shortest_paths(v1, [v3, v2])
    vertices_1, edges_1 = paths[0]
    vertices_2, edges_2 = paths[1]
    assert vertices_1 == [v1, v2 ,v3]
    assert edges_1 == [e1, e2]
    assert vertices_2 == [v1, v2]
    assert edges_2 == [e1]

def test_edge_endpoints(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "node1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node1"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "node2",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node2"]
    })
    edge = graph.add_edge(v1, v2, {
        "sentence": "node1 is connected to node2",
        "entity1": "node1",
        "entity2": "node2",
        "terminal": False
    })
    source, target = graph.edge_endpoints(edge)
    assert source == v1
    assert target == v2

def test_get_neighbors(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "node1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node1"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "node2",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node2"]
    })
    v3 = graph.add_vertex({
        "id": "3",
        "label": "node3",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node3"]
    })
    v4 = graph.add_vertex({
        "id": "4",
        "label": "node4",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node4"]
    })
    graph.add_edge(v1, v2, {
        "sentence": "node1 is connected to node2",
        "entity1": "node1",
        "entity2": "node2",
        "terminal": False
    })
    graph.add_edge(v1, v3, {
        "sentence": "node1 is connected to node3",
        "entity1": "node1",
        "entity2": "node3",
        "terminal": False
    })
    graph.add_edge(v2, v4, {
        "sentence": "node1 is connected to node4",
        "entity1": "node1",
        "entity2": "node4",
        "terminal": False
    })
    neighbors = list(graph.get_neighbors(v1))
    assert len(neighbors) == 2
    assert set(neighbors) == {v2, v3}

def test_serialization(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "node1",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node1"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "node2",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["node2"]
    })
    graph.add_edge(v1, v2, {
        "sentence": "node1 is connected to node2",
        "entity1": "node1",
        "entity2": "node2",
        "terminal": False
    })
    
    # Test serialization/deserialization
    data = graph.to_bytes()
    new_graph = IGraphSentenceGraph.from_bytes(data)
    assert new_graph.num_vertices() == graph.num_vertices()
    assert new_graph.num_edges() == graph.num_edges()

def test_path_length(graph):
    v1 = graph.add_vertex({
        "id": "1",
        "label": "start",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["start"]
    })
    v2 = graph.add_vertex({
        "id": "2",
        "label": "middle",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["middle"]
    })
    v3 = graph.add_vertex({
        "id": "3",
        "label": "end",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["end"]
    })
    v4 = graph.add_vertex({
        "id": "4",
        "label": "alternate",
        "terminal": True,
        "ner_label": "PERSON",
        "aliases": ["alternate"]
    })
    e1 = graph.add_edge(v1, v2, {
        "sentence": "start is connected to middle",
        "entity1": "start",
        "entity2": "middle",
        "terminal": False
    })
    e2 = graph.add_edge(v2, v3, {
        "sentence": "middle is connected to end",
        "entity1": "middle",
        "entity2": "end",
        "terminal": False
    })
    e3 = graph.add_edge(v1, v4, {
        "sentence": "start is connected to alternate",
        "entity1": "start",
        "entity2": "alternate",
        "terminal": False
    })
    e4 = graph.add_edge(v4, v3, {
        "sentence": "alternate is connected to end",
        "entity1": "alternate",
        "entity2": "end",
        "terminal": False
    })
    graph.set_edge_weights([1.0, 2.0, 1.5, 0.5])
    path1 = ([v1, v2, v3], [e1, e2])
    path2 = ([v1, v4, v3], [e3, e4])
    assert graph.path_length(path1) == 3.0
    assert graph.path_length(path2) == 2.0
