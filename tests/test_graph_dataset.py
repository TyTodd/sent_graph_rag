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
        prop_noun_node = graph.get_vertex(prop_noun_text)
            
        
        
        
        
        
        
        
def test_graph_dataset(load_model):
    '''
    1. make test.json -- add dummy info using 1 of guide.json
        needs 20 rows (dicts)
    2. Make Dataset reader -- override init read and len 
    3. Test:
        graph is correct, have output dataset and check 
        each row of each together checking 
        function in Graph class that will tak



        (1) Take context run through language model (load_model)
            docs = load_model.spacy_model.nlp.pipe([context, question, answers])
            graph = SentenceGraph.from_doc(docs[0], "igraph") --> graph 1

            Check to make sure graph matches the graph (2) produced
            Make sure answers is answers and questions is questions -- all of them
                question_entities is docs[1].ents --> have to convert to strings (currently a list of span Objects) use .text
                question_embedding: 
                with load_model.embedding_model as em:
                    em.get_emeddings([question])

    for embeddings -- need to be within a certain range (0.001)

    '''
    # Dataset Reader --- DatasetReader()
    
    with open("test.json", "r") as f:
        text = f.read()
    docs = load_model.spacy_model.nlp.pipe([context, question, answers])
    graph = SentenceGraph.from_doc(docs[0], "igraph")
    pass
    
    
    