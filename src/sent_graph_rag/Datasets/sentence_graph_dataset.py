# import spacy
# from spacy.pipeline import EntityLinker
# from fastcoref import spacy_component
import warnings
from spacy.kb import InMemoryLookupKB
import numpy as np
import time
import logging
from disjoint_set import DisjointSet
from .readers import DatasetReader
import json
from tqdm import tqdm
from fastavro import writer
from fastavro import reader
import spacy
from datasets.utils.logging import disable_progress_bar
import os
from .embedder import get_qas_entities, add_embeddings_to_graphs, get_query_embeddings
from typing import List, Tuple, Dict, Union, Optional, Iterator, Literal
from .language_models import LanguageModel, EmbeddingModel, SpacyModel
from io import BytesIO
import shutil
from .graph import GraphToolSentenceGraph, IGraphSentenceGraph

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
    def __init__(self, path_name: str, language_model: LanguageModel = None, verbose: bool = False, graph_type: Literal["igraph", "graph-tool"] = "igraph", directory_mode: bool = False, metadata: dict = {}):
        self.language_model = language_model
        self.directory_mode = directory_mode
        self.entity_vector_length = 64
        # self.kb = InMemoryLookupKB(vocab=self.language_model.nlp.vocab, entity_vector_length=self.entity_vector_length)
        self.verbose = verbose
        self.path_name = path_name
        self.data_length = None
        self.graph_type = graph_type
        self.metadata = metadata

        

    # def reset_kb(self):
    #     self.kb = InMemoryLookupKB(vocab=self.language_model.nlp.vocab, entity_vector_length=self.entity_vector_length)
    
    @staticmethod
    def from_graph_dataset(in_name: str, verbose = False):
        if os.path.isdir(in_name):
            with open(os.path.join(in_name, "metadata.json"), "r") as f:
                metadata = json.load(f)
                graph_type = metadata["graph_type"]
            sentence_graph_dataset = SentenceGraphDataset(path_name = in_name, verbose = verbose, graph_type = graph_type, directory_mode = True, metadata = metadata)
        else:
            with open(in_name, "rb") as f:
                metadata = reader(f).metadata
                graph_type = metadata["graph_type"]
            sentence_graph_dataset = SentenceGraphDataset(path_name = in_name, verbose = verbose, graph_type = graph_type, directory_mode = False, metadata = metadata)
        return sentence_graph_dataset
        
            
    def __iter__(self):
        if self.directory_mode:
            with open(self.path_name, "rb") as fo:
                for record in reader(fo):
                    yield record
        else:
            for file in os.listdir(self.path_name):
                with open(os.path.join(self.path_name, file), "rb") as fo:
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
        Graph = IGraphSentenceGraph if self.graph_type == "igraph" else GraphToolSentenceGraph
        graphs = []
        processed_graphs = 0
        all_questions = [] # all questions for each graph in a chunk (flattened)
        all_qas = [] # qas for each graph in the chunk (unflattened)
        with tqdm(total=self.data_length, desc="Converting dataset") as pbar:
            with self.language_model.embedding_model as em:
                # with open(in_path, "rb") as fo:
                for index, record in enumerate(self.read_temp_dataset(in_path)):
                    print(record["qas"])
                    graphs.append(Graph.from_bytes(record['graph']))
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
                                "graph": graphs.pop(0).to_bytes(),
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
                                "graph": graphs.pop(0)[0].to_bytes(),
                                "qas": [all_qas.pop(0) for _ in range(num_qas)]
                                }
                            pbar.update(1)
                        
                        texts = []
        
        
    def create_graphs(self, texts, spacy_model: SpacyModel, verbose = False):
        inference_start = time.time()
        docs = list(spacy_model.nlp.pipe(texts))
        if self.verbose: print(f"Inference time: {time.time() - inference_start}")

        graphs = []
        for doc in docs:
            graphs.append(self.create_graph_from_doc(doc, verbose = verbose))
            # self.reset_kb()


        return graphs


    def create_graph_from_doc(self, doc, verbose = False):
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

        graph, id_to_vertex = self.data_to_graph(eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = self.verbose)

        if self.verbose: print(f"Graph Build Time: {time.time() - graph_build_start}")
        return graph, id_to_vertex

    def data_to_graph(self, eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = False):
        id_to_vertex = {}
        graph = None
        if self.graph_type == "igraph":
            graph = IGraphSentenceGraph(doc.text)
        elif self.graph_type == "graph-tool":
            graph = GraphToolSentenceGraph(doc.text)

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
                    v1 = graph.add_vertex({"id": entity_id1, "label": eid_name_map[entity_id1], "terminal": False, "ner_label": eid_label_map[entity_id1], "aliases": list(eid_alias_map[entity_id1])})
                    id_to_vertex[entity_id1] = v1

                sentence = reference1.sent
                num_diff_entities = 0
                for reference2_loc in sentence_ref_map[sentence]:
                    reference2 = doc.char_span(reference2_loc.start_char, reference2_loc.end_char, alignment_mode="contract")
                    entity_id2 = reference_eid_map[reference2]

                    if entity_id1 != entity_id2:
                        num_diff_entities += 1

                    if entity_id2 not in id_to_vertex: # if we haven't created a vertex for this entity_id yet create one
                        v2 = graph.add_vertex({"id": entity_id2, "label": eid_name_map[entity_id2], "terminal": False, "ner_label": eid_label_map[entity_id2], "aliases": list(eid_alias_map[entity_id2])})
                        id_to_vertex[entity_id2] = v2

                    edge_hash = frozenset([sentence.text, entity_id1, entity_id2])
                    if reference1.start < reference2.start and entity_id1 != entity_id2 and edge_hash not in added_edges: #only add edge if reference1 comes before reference2 so we don't have duplicate edges
                        sentence_offset = sentence.start_char
                        ent1_location = (reference1.start_char - sentence_offset, reference1.end_char - sentence_offset)
                        ent2_location = (reference2.start_char - sentence_offset, reference2.end_char - sentence_offset)

                        v1 = id_to_vertex[entity_id1]
                        v2 = id_to_vertex[entity_id2]
                        e1 = graph.add_edge(v1, v2, {"sentence": sentence.text, "entity1": list(ent1_location), "entity2": list(ent2_location), "terminal": False})

                        pairs.setdefault(frozenset([entity_id1, entity_id2]), 0)
                        pairs[frozenset([entity_id1, entity_id2])] += 1
                        added_edges.add(edge_hash)
                if num_diff_entities < 1: # draw edge from entity to terminal node
                    terminal_id = graph.vertex_properties["id"][v1] + "_TERMINAL"
                    if terminal_id not in id_to_vertex:
                        v2 = graph.add_vertex({"id": terminal_id, "label": "terminal_node", "terminal": True, "ner_label": "none", "aliases": ["terminal_node"]})
                        id_to_vertex[terminal_id] = v2

                    v2 = id_to_vertex[terminal_id]

                    sentence_offset = sentence.start_char
                    ent1_location = (reference1.start_char - sentence_offset, reference1.end_char - sentence_offset)
                    e1 = graph.add_edge(v1, v2, {"sentence": sentence.text, "entity1": list(ent1_location), "entity2": list(ent1_location), "terminal": True})
        return graph, id_to_vertex


    def create_entity_vector(self, text):
        # TODO: Replace this with a model that generates embeddings for entities
        return np.random.rand(self.entity_vector_length)



    