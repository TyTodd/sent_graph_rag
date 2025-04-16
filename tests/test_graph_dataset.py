import pytest
from sent_graph_rag import LanguageModel, SentenceGraph

@pytest.fixture(scope="session")
def load_model():
    language_model = LanguageModel()
    return language_model


def is_prop_noun(ref) -> bool:
    pos = [tok.pos_ for tok in ref]
    return "PROPN" in pos
    
    
    
def test_mit_graph_build(load_model):
    with open("mit_wiki.txt", "r") as f:
        text = f.read()
    doc = load_model.spacy_model.nlp.pipe([text])[0]
    graph = SentenceGraph.from_doc(doc, "igraph")
    # check that each entity is in an edge in the graph and that edge is connected to a node that has an alias that is the entity
    for entity in doc.ents:
        entity_text = entity.text
        ent_is_alias = lambda x: entity_text in x
        ent_has_label = lambda x: x == entity.label_
        # check if entity is mentioned in the graph as an alias of a node with the same label
        graph.set_vertex_filter("aliases" , ent_is_alias)
        graph.set_vertex_filter("label" , ent_has_label)
        nodes = [graph.iter_vertices()]
        graph.clear_filters()
        assert len(nodes) > 0, f"Entity '{entity_text}' not found in graph"
        assert len(nodes) < 2, f"Entity '{entity_text}' found in multiple nodes in graph"
        
        # check that the node has (as an edge) every sentence that mentions the entity
        node = nodes[0]
        sentence = entity.sent.text
        edges = graph.get_edges(node)
        edge_sentences = [graph.get_edge_property(edge, "sentence") for edge in edges]
        assert sentence in edge_sentences, f"""Sentence for entity '{entity_text}' not connected to node: '{graph.get_vertex_property(node, "id")}'
        sentence: {sentence}
        edges: {edge_sentences}
        """
    
    for cluster in doc._.coref_clusters:
        prop_noun = next((ref for ref in cluster if is_prop_noun(ref)), None) # get first reference in cluster that is a proper noun
        if prop_noun is None: # if no proper noun in cluster then it was never added to the graph
            continue
        prop_noun_text = prop_noun.text
        nodes = graph.set_vertex_filter("aliases", filter_fn=lambda x: prop_noun_text in x)
        assert len(nodes) > 0, f"Entity '{prop_noun_text}' not found in graph"
        assert len(nodes) < 2, f"Entity '{prop_noun_text}' found in multiple nodes in graph"
        
        
        
        
        
        
    
    
    
    