import pytest
from sent_graph_rag import LanguageModel, SentenceGraph, IGraphSentenceGraph
from sent_graph_rag.Datasets import DatasetReader

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
        
        # check that the node has (as an edge) a sentence that mentions the entity
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
        # check if cluster's proper noun is mentioned in the graph as an alias of a node with the same label
        graph.set_vertex_filter("aliases", filter_fn=lambda x: prop_noun_text in x)
        graph.set_vertex_filter("label" , lambda x: x == prop_noun.label_)
        nodes = [graph.iter_vertices()]
        graph.clear_filters()
        assert len(nodes) > 0, f"Entity '{prop_noun_text}' not found in graph"
        assert len(nodes) < 2, f"Entity '{prop_noun_text}' found in multiple nodes in graph"
        
        # check that the node has (as an edge) a sentence that mentions each reference in the cluster
        prop_noun_node = nodes[0]
        edges = graph.get_edges(prop_noun_node)
        edge_sentences = [graph.get_edge_property(edge, "sentence") for edge in edges]
        for ref in cluster:
            ref_sentence = ref.sent.text
        
        
        
        
        
        
class TestReader(DatasetReader):
    """
    Reader for the test dataset.
    """
    def __init__(self, file_path: str):
        super().__init__(file_path)
        with open(file_path, 'r') as f:
            self.test_data = json.load(f)
        self.data_length = len(self.test_data)
        
    def read(self) -> Iterator[Row]:
        for row in self.test_data:
            yield {"context": row["context"], "qas": row["qas"]}
    
    def __len__(self) -> int:
        return self.data_length
            
        
def test_graph_dataset(load_model):
    # Test dataset
    test_reader = TestReader("test.json")
    graph_dataset = SentenceGraphDataset.from_dataset(test_reader, out_path="test_graph_datset.avro", graph_type="igraph", language_model=load_model, chunk_size=5, overwrite=True)

    with open("test.json", "r") as f:
        test_data = f.read()
    
    # Checking if correct (test.json) turns into test (graph1)
    for correct, test in zip(test_data, graph_dataset):
        # Checking graphs
        context = correct["context"]
        context_doc = load_model.spacy_model.nlp.pipe([context])[0]
        graph = SentenceGraph.from_doc(context_doc, "igraph")
        assert graph == IGraphSentenceGraph.from_bytes(test["graph"]), "graphs don't match"
        
        questions = [qa["question"] for qa in correct["qas"]]
        question_docs = load_model.spacy_model.nlp.pipe(questions)
        with load_model.embedding_model as em:
            question_embeddings = em.get_emeddings(questions).tolist()
        
        for index, qa in enumerate(correct["qas"]):
            # Checking answers
            answers = qa["answers"]
            assert answers == test["qas"][index]["answers"], "answers don't match"

            # Checking answer entities 
            answer_docs = load_model.spacy_model.nlp.pipe(answers)
            answer_ents = [ent.text for answer_doc in answer_docs for ent in answer_doc.ents]
            assert answer_ents == test["qas"][index]["answer_entities"], "answer_entities don't match"

            # Checking questions
            question = qa["question"]
            assert question == test["qas"][index]["question"], "questions don't match"
            
            # Checking question entities
            question_ents = [ent.text for ent in question_docs[index].ents]
            assert question_ents == test["qas"][index]["question_entities"], "question_entities don't match"
            
            # Checking question embeddings
            assert question_embeddings[index] == test["qas"][index]["question_embedding"], "question_embeddings don't match"
