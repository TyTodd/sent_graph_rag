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
from .graph import GraphToolSentenceGraph, IGraphSentenceGraph, SentenceGraph

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
            graphs.append(SentenceGraph.from_doc(doc, self.graph_type))

        return graphs

    def create_entity_vector(self, text):
        # TODO: Replace this with a model that generates embeddings for entities
        return np.random.rand(self.entity_vector_length)



    