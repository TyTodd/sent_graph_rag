# import spacy
# from spacy.pipeline import EntityLinker
# from fastcoref import spacy_component
import warnings
from spacy.kb import InMemoryLookupKB
import numpy as np
import networkx as nx
import time
import graph_tool as gt
import logging
from disjoint_set import DisjointSet
from .readers import DatasetReader
import json
from tqdm import tqdm
from ..functions import graph_to_string
from fastavro import writer
from fastavro import reader
import spacy
from datasets.utils.logging import disable_progress_bar
import os
from .embedder import get_qas_entities, add_embeddings_to_graphs, get_query_embeddings
from typing import List, Tuple, Dict, Union, Optional, Iterator
from .language_models import LanguageModel, EmbeddingModel, SpacyModel
from io import BytesIO

temp_dataset_schema = {
    "type": "record",
    "name": "Dataset",
    "fields": [
        {"name": "graph", "type": "bytes"},
        {
            "name": "qas",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "QAItem",
                    "fields": [
                        {"name": "question", "type": "string"},
                        {"name": "question_entities", "type": {"type": "array", "items": "string"}},
                        {"name": "answers", "type": {"type": "array", "items": "string"}},
                        {"name": "answer_entities", "type": {"type": "array", "items": "string"}}
                    ]
                }
            }
        }
    ]
}

graph_dataset_schema = {
    "type": "record",
    "name": "Dataset",
    "fields": [
        {"name": "graph", "type": "bytes"},
        {
            "name": "qas",
            "type": {
                "type": "array",
                "items": {
                    "type": "record",
                    "name": "QAItem",
                    "fields": [
                        {"name": "question", "type": "string"},
                        {"name": "question_embedding", "type": {"type": "array", "items": "float"}},
                        {"name": "question_entities", "type": {"type": "array", "items": "string"}},
                        {"name": "answers", "type": {"type": "array", "items": "string"}},
                        {"name": "answer_embedding", "type": {"type": "array", "items": "float"}},
                        {"name": "answer_entities", "type": {"type": "array", "items": "string"}}
                    ]
                }
            }
        }
    ]
}

def get_records_size(records: List[Dict], schema: Dict) -> int:
    buffer = BytesIO()
    writer(buffer, schema, records)
    size = len(buffer.getvalue())
    return size
    
    
class SentenceGraphDataset:
    def __init__(self, language_model: LanguageModel = None, verbose: bool = False):
        if language_model is None:
            language_model = LanguageModel()
        self.language_model = language_model
        
        self.entity_vector_length = 64
        # self.kb = InMemoryLookupKB(vocab=self.language_model.nlp.vocab, entity_vector_length=self.entity_vector_length)
        self.verbose = verbose
        self.path_name = None
        self.data_length = None
        
        

    # def reset_kb(self):
    #     self.kb = InMemoryLookupKB(vocab=self.language_model.nlp.vocab, entity_vector_length=self.entity_vector_length)
    
    @staticmethod
    def from_graph_dataset(in_name: str, nlp = None, verbose = False):
        sentence_graph_dataset = SentenceGraphDataset(nlp, verbose)
        sentence_graph_dataset.path_name = in_name
        
    
    @staticmethod
    def from_dataset(datatset_reader: DatasetReader,  out_dir: str = None, out_path: str = None, language_model: LanguageModel = None, chunk_size: int = 1000, metadata: dict = {}, verbose:bool = False, max_file_size: int = None, overwrite: bool = False):
        """
        Convert a dataset to a sentence graph dataset.
        datatset_reader: DatasetReader class for the dataset to convert
        out_dir: str path to save the sentence graph dataset (can only be set if out_path is None and max_file_size is set)
        out_path: str path to save the sentence graph dataset (can only be set if out_dir is None and max_file_size is None)
        language_model: LanguageModel language model for inference 
        chunk_size: int number of rows to process at a time
        metadata: dict metadata to save with the dataset
        verbose: bool 
        max_file_size: int maximum size of the file to save in GB
        """
        if (out_dir is None) == (out_path is None):
            raise ValueError("Specify exactly one of `out_dir` or `out_path`.")

        if out_dir and max_file_size is None:
            raise ValueError("`max_file_size` must be set when using `out_dir`.")
        
        if out_path and max_file_size is None:
            warnings.warn(
                "`max_file_size` is set but you are using `out_path` so file sizes will not be limited. Use `out_dir` instead to split files.",
                UserWarning
            )
        
        if out_dir and os.path.exists(out_dir) and os.listdir(out_dir) and not overwrite:
            response = input(f"Output directory {out_dir} is not empty. Would you like to empty it? [y/N]: ")
            if response.lower() != 'y' or response.lower() != '':
                raise ValueError("Aborting: Output directory is not empty.")
            shutil.rmtree(out_dir)
            
        sentence_graph_dataset = SentenceGraphDataset(language_model, verbose)
        sentence_graph_dataset.data_length = len(datatset_reader)
        
        if out_path:
            temp_path = os.path.splitext(out_path)[0] + "_temp" + os.path.splitext(out_path)[1]
            with open(temp_path, "wb") as out:
                writer(out, temp_dataset_schema, sentence_graph_dataset.temp_dataset(datatset_reader, chunk_size))
                
            with open(out_path, "wb") as out:
                writer(out, graph_dataset_schema, sentence_graph_dataset.embeded_dataset(temp_path, chunk_size))
            
            os.remove(temp_path)
        else:
            temp_dataset_iter = self.temp_dataset(datatset_reader, chunk_size)
            for i, dataset_generator in enumerate(self.dataset_split(temp_dataset_iter, max_file_size, chunk_size)):
                temp_dir = out_dir + "/temp"
                os.makedirs(temp_dir)
                with open(temp_dir + f"/temp_{i}.avro", "wb") as out:
                    writer(out, temp_dataset_schema, dataset_generator)
                
            
            embeded_dataset_iter = self.embeded_dataset(temp_dir, chunk_size)
            file_name = datatset_reader.get_name()
            for i, dataset_generator in enumerate(self.dataset_split(embeded_dataset_iter, max_file_size, chunk_size)):
                with open(out_dir + f"/{file_name}_{i}.avro", "wb") as out:
                    writer(out, graph_dataset_schema, dataset_generator)
            
            shutil.rmtree(temp_dir)
            
        return sentence_graph_dataset
            
    def __iter__(self):
        with open(self.path_name, "rb") as fo:
            for record in reader(fo):
                yield record
    
  
    def read_temp_dataset(self, in_path: str):
        if os.path.isfile(in_path):
            with open(in_path, "rb") as fo:
                for record in reader(fo):
                    yield record
        else:
            for file in os.listdir(in_path):
                with open(os.path.join(in_path, file), "rb") as fo:
                    for record in reader(fo):
                        yield record
        
    def embeded_dataset(self, in_path: str, chunk_size: int = 1000):
        graphs = []
        processed_graphs = 0
        all_questions = [] # all questions for each graph in a chunk (flattened)
        all_qas = [] # qas for each graph in the chunk (unflattened)
        with tqdm(total=self.data_length, desc="Converting dataset") as pbar:
            with self.language_model.embedding_model as em:
                # with open(in_path, "rb") as fo:
                for index, record in enumerate(self.read_temp_dataset(in_path)):
                    print(record["qas"])
                    graphs.append(string_to_graph(record['graph']))
                    processed_graphs += 1
                    questions = [qas['question'] for qas in record['qas']]
                    all_questions.extend(questions)
                    all_qas.append(record['qas'])
                    if len(graphs) >= chunk_size or (processed_graphs >= self.data_length - 1 and len(graphs) > 0):
                        add_embeddings_to_graphs(graphs, em)
                        query_embeddings = get_query_embeddings(all_questions, em)
                        while graphs:
                            qas = all_qas.pop(0)
                            for qa in qas:
                                qa['question_embedding'] = query_embeddings.pop(0)
                            yield {
                                "graph": graph_to_string(graphs.pop(0)[0]),
                                "qas": qas
                            }
                            pbar.update(1)
                        all_questions = []
                            
    
    def dataset_split(self, dataset_iter: Iterator, max_file_size: int, chunk_size: int = 1000):
        # dataset_iter = self.temp_dataset(dataset_reader, chunk_size)  # Shared across all splits

        def make_dataset_generator():
            curr_size = 0
            seen_records = []

            def dataset_generator():
                nonlocal curr_size, seen_records
                for record in dataset_iter:
                    yield record

                    seen_records.append(record)
                    if len(seen_records) >= 100:
                        curr_size += get_records_size(seen_records, temp_dataset_schema)
                        seen_records = []

                    if curr_size >= max_file_size:
                        break

            return dataset_generator

        while True:
            g = make_dataset_generator()
            try:
                peek = next(g())
            except StopIteration:
                break

            def wrapped_generator(first_record=peek, gen=g):
                yield first_record
                yield from gen()

            yield wrapped_generator
  
            
    def temp_dataset(self, dataset_reader: DatasetReader, chunk_size: int = 1000):
        texts = [] # texts for all graphs in the chunk
        all_qas = [] # qas for all graphs in the chunk (flattened)
        all_num_qas = [] # number of qas for each graph
        processed_graphs = 0
        with tqdm(total=self.data_length, desc="Converting dataset") as pbar:
            with self.language_model.spacy_model as spacy_model:
                for row in dataset_reader.read():
                    texts.append(row['context'])
                    all_qas.append(row['qas'])
                    all_num_qas.append(len(row['qas']))
                    processed_graphs += 1
                    if len(texts) >= chunk_size or (processed_graphs >= self.data_length - 1 and len(texts) > 0):
                        graphs = self.create_graphs(texts, spacy_model)
                        num_qas = all_num_qas.pop(0)
                        while graphs:
                            yield {
                                "graph": graph_to_string(graphs.pop(0)[0]),
                                "qas": [all_qas.pop(0) for _ in range(num_qas)]
                                }
                            pbar.update(1)
                        
                        texts = []
        
        
    def create_graphs(self, texts, spacy_model: SpacyModel, graph_type = "gt", verbose = False):
        inference_start = time.time()
        docs = list(spacy_model.nlp.pipe(texts))
        if self.verbose: print(f"Inference time: {time.time() - inference_start}")

        graphs = []
        for doc in docs:
            graphs.append(self.create_graph_from_doc(doc, graph_type = graph_type, verbose = verbose))
            # self.reset_kb()


        return graphs


    def create_graph_from_doc(self, doc, graph_type = "nx", verbose = False):
        graph_build_start = time.time()
        entities = set()
        for ent in doc.ents:
            entities.add(ent)
        #Map each entity to the cluster that contains it
        entity_cluster_map = {}
        added = set()
        eid_reference_map = {}
        for cluster in doc._.coref_clusters:
            for loc in cluster:
                # ent = doc.char_span(loc[0], loc[1], alignment_mode="contract")
                entity_cluster_map[loc] = cluster
        # Create a dictionary mapping unique entities names to all of their mentions across the doc
        visited_references = set()
        unique_entities = {}
        eid_name_map = {} #mapping entity ids to the name of the entiy
        eid_label_map = {} #mapping entity ids to the label of the entity
        for ent in doc.ents:
                # unique id will be {entityText}_{label}
                ent_loc = (ent.start_char, ent.end_char)
                if ent_loc not in visited_references:
                    entity_id = f"{ent.text}_{ent.label_}"
                    eid_name_map[entity_id] = ent.text
                    eid_label_map[entity_id] = ent.label_

                    unique_entities.setdefault(entity_id, [])

                    if ent_loc in entity_cluster_map: # if in a cluster map to all references of entity
                        visited_references.update(set(entity_cluster_map[ent_loc]))
                        unique_entities[entity_id].extend(entity_cluster_map[ent_loc])
                    else: # if not in cluster map to just the single mention of entity
                        unique_entities[entity_id].append((ent.start_char, ent.end_char))
                        visited_references.add(ent_loc)
        # For each entity, add it's aliases (names of other entities in cluster) to the Knowledge Base
        reference_eid_map = {} # Also create mapping of each reference to its unique entity id
        sentence_ref_map = {} # Also Create a mapping of sentences to references/entities
        eid_alias_map = {} #also create mapping of each eid to its aliases
        for entity_id in unique_entities:
            # self.kb.add_entity(entity=entity_id, freq=100, entity_vector=self.create_entity_vector(eid_name_map[entity_id]))
            added = set()
            eid_alias_map.setdefault(entity_id, set([eid_name_map[entity_id]]))
            for reference_loc in unique_entities[entity_id]:
                reference = doc.char_span(reference_loc[0], reference_loc[1], alignment_mode="contract")
                reference_eid_map[reference] = entity_id
                sentence_ref_map.setdefault(reference.sent, [])
                sentence_ref_map[reference.sent].append(reference)
                #add reference to KB if it is a named entity and we haven't added it yet
                if (reference.text not in added) and (reference in entities):
                    added.add(reference.text)
                    # self.kb.add_alias(alias=reference.text, entities=[entity_id], probabilities=[1.0])

                pos = [tok.pos_ for tok in reference]
                # if entity_id == "The Massachusetts Institute of Technology_ORG":
                #   print(reference.text)
                if "PROPN" in pos:
                    eid_alias_map[entity_id].add(reference.text)

        if graph_type == "nx":
            graph, id_to_vertex = self.data_to_nx(eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = self.verbose)
        else:
            graph, id_to_vertex = self.data_to_gt(eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = self.verbose)

        if self.verbose: print(f"Graph Build Time: {time.time() - graph_build_start}")
        return graph, id_to_vertex

    def data_to_gt(self, eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = False):
        id_to_vertex = {}
        graph = gt.Graph(directed=False)
        graph.graph_properties["corpus"] = graph.new_graph_property("string")
        graph.graph_properties["corpus"] = doc.text

        graph.edge_properties["sentence"] = graph.new_edge_property("string")  # For the sentence
        graph.edge_properties["entity1"] = graph.new_edge_property("vector<int>")  # For entity1 location
        graph.edge_properties["entity2"] = graph.new_edge_property("vector<int>")  # For entity2 location
        graph.edge_properties["terminal"] = graph.new_edge_property("bool")

        graph.vertex_properties["id"] = graph.new_vertex_property("string")
        graph.vertex_properties["label"] = graph.new_vertex_property("string")
        graph.vertex_properties["terminal"] = graph.new_vertex_property("bool")
        graph.vertex_properties["ner_label"] = graph.new_vertex_property("string")
        graph.vertex_properties["aliases"] = graph.new_vertex_property("object")
        added_edges = set()
        #TODO: Find a way to not have to loop through every edge twice.
        # Now build graph
        pairs = {}
        for entity_id in unique_entities:
            #iterate through each reference and draw sentence edges between all entities in sentence
            for reference1_loc in unique_entities[entity_id]:
                reference1 = doc.char_span(reference1_loc[0], reference1_loc[1], alignment_mode="contract")
                entity_id1 = reference_eid_map[reference1]
                if entity_id1 not in id_to_vertex: # if we havents created a vertex for this entity_id yet create one
                    v1 = graph.add_vertex()
                    graph.vertex_properties["id"][v1] = entity_id1
                    graph.vertex_properties["label"][v1] = eid_name_map[entity_id1]
                    graph.vertex_properties["terminal"][v1] = False
                    graph.vertex_properties["ner_label"][v1] = eid_label_map[entity_id1]
                    graph.vertex_properties["aliases"][v1] = list(eid_alias_map[entity_id1])
                    id_to_vertex[entity_id1] = v1

                sentence = reference1.sent
                num_diff_entities = 0
                for reference2_loc in sentence_ref_map[sentence]:
                    reference2 = doc.char_span(reference2_loc.start_char, reference2_loc.end_char, alignment_mode="contract")
                    entity_id2 = reference_eid_map[reference2]

                    if entity_id1 != entity_id2:
                        num_diff_entities += 1

                    if entity_id2 not in id_to_vertex: # if we haven't created a vertex for this entity_id yet create one
                        v2 = graph.add_vertex()
                        graph.vertex_properties["id"][v2] = entity_id2
                        graph.vertex_properties["label"][v2] = eid_name_map[entity_id2]
                        graph.vertex_properties["terminal"][v2] = False
                        graph.vertex_properties["ner_label"][v2] = eid_label_map[entity_id2]
                        graph.vertex_properties["aliases"][v2] = list(eid_alias_map[entity_id2])
                        id_to_vertex[entity_id2] = v2

                    edge_hash = frozenset([sentence.text, entity_id1, entity_id2])
                    if reference1.start < reference2.start and entity_id1 != entity_id2 and edge_hash not in added_edges: #only add edge if reference1 comes before reference2 so we don't have duplicate edges
                        sentence_offset = sentence.start_char
                        ent1_location = (reference1.start_char - sentence_offset, reference1.end_char - sentence_offset)
                        ent2_location = (reference2.start_char - sentence_offset, reference2.end_char - sentence_offset)

                        v1 = id_to_vertex[entity_id1]
                        v2 = id_to_vertex[entity_id2]
                        e1 = graph.add_edge(v1, v2)
                        graph.edge_properties["sentence"][e1] = sentence.text
                        graph.edge_properties["entity1"][e1] = list(ent1_location)
                        graph.edge_properties["entity2"][e1] = list(ent2_location)
                        graph.edge_properties["terminal"][e1] = False


                        pairs.setdefault(frozenset([entity_id1, entity_id2]), 0)
                        pairs[frozenset([entity_id1, entity_id2])] += 1
                        added_edges.add(edge_hash)
                if num_diff_entities < 1: # draw edge from entity to terminal node
                    terminal_id = graph.vertex_properties["id"][v1] + "_TERMINAL"
                    if terminal_id not in id_to_vertex:
                        v2 = graph.add_vertex()
                        graph.vertex_properties["id"][v2] = terminal_id
                        graph.vertex_properties["label"][v2] = "terminal_node"
                        graph.vertex_properties["terminal"][v2] = True
                        graph.vertex_properties["ner_label"][v2] = "none"
                        graph.vertex_properties["aliases"][v2] = ["terminal_node"]
                        id_to_vertex[terminal_id] = v2

                    v2 = id_to_vertex[terminal_id]

                    sentence_offset = sentence.start_char
                    ent1_location = (reference1.start_char - sentence_offset, reference1.end_char - sentence_offset)
                    e1 = graph.add_edge(v1, v2)
                    graph.edge_properties["sentence"][e1] = sentence.text
                    graph.edge_properties["entity1"][e1] = list(ent1_location)
                    graph.edge_properties["entity2"][e1] = list(ent1_location)
                    graph.edge_properties["terminal"][e1] = True

        return graph, id_to_vertex


    def data_to_nx(self, eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = False):
        id_to_vertex = {}
        graph = nx.MultiGraph()
        added = set()
        added_edges = set()
        #TODO: Find a way to not have to loop through every edge twice.
        # Now build graph
        pairs = {}
        for entity_id in unique_entities:
            #iterate through each reference and draw sentence edges between all entities in sentence
            for reference1_loc in unique_entities[entity_id]:
                reference1 = doc.char_span(reference1_loc[0], reference1_loc[1], alignment_mode="contract")
                entity_id1 = reference_eid_map[reference1]
                if entity_id1 not in added: # if we havents created a vertex for this entity_id yet create one
                    added.add(entity_id1)
                    graph.add_node(entity_id1, label=eid_name_map[entity_id1], ner_label = eid_label_map[entity_id1], terminal = False, aliases = eid_alias_map[entity_id1]) #networkx

                sentence = reference1.sent

                if len(sentence_ref_map[sentence]) > 1:

                    for reference2_loc in sentence_ref_map[sentence]:
                        reference2 = doc.char_span(reference2_loc.start_char, reference2_loc.end_char, alignment_mode="contract")
                        entity_id2 = reference_eid_map[reference2]
                        if entity_id2 not in added: # if we haven't created a vertex for this entity_id yet create one
                            added.add(entity_id2)
                            graph.add_node(entity_id2, label=eid_name_map[entity_id2], ner_label = eid_label_map[entity_id2], terminal = False, aliases = eid_alias_map[entity_id2]) #networkx

                        edge_hash = frozenset([sentence.text, entity_id1, entity_id2])
                        if reference1.start < reference2.start and entity_id1 != entity_id2 and edge_hash not in added_edges: #only add edge if reference1 comes before reference2 so we don't have duplicate edges
                            sentence_offset = sentence.start_char
                            ent1_location = (reference1.start_char - sentence_offset, reference1.end_char - sentence_offset)
                            ent2_location = (reference2.start_char - sentence_offset, reference2.end_char - sentence_offset)
                            #networkx
                            graph.add_edge(entity_id1,
                                                        entity_id2,
                                                        sentence=sentence.text,
                                                        entity_spans = {entity_id1: ent1_location, entity_id2: ent2_location},
                                                        terminal = False
                                                        )

                            pairs.setdefault(frozenset([entity_id1, entity_id2]), 0)
                            pairs[frozenset([entity_id1, entity_id2])] += 1
                            added_edges.add(edge_hash)
                    else: # draw edge from entity to terminal node
                        terminal_id = entity_id1 + "_TERMINAL"
                        if terminal_id not in added:
                            graph.add_node(terminal_id, label='terminal_node', ner_label = 'none', terminal = True)
                            added.add(terminal_id)
                        else:
                            graph.add_edge(entity_id1,
                                                        entity_id2,
                                                        sentence=sentence.text,
                                                        entity_spans = {entity_id1: ent1_location, terminal_id: ent1_location},
                                                        terminal = True
                                                        )
        return graph, None



    def create_entity_vector(self, text):
        # TODO: Replace this with a model that generates embeddings for entities
        return np.random.rand(self.entity_vector_length)



    