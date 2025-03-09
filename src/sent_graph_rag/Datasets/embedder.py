import spacy
from spacy.language import Language
import torch
import gc
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
import time

# from spacy_transformers import TransformersLanguage
class EmbeddingModel(ABC):
    def __init__(self, embedding_model, dim):
        """
        Intializes Embedding model
        embedding_model: Embedding model
        tokenizer: Tokenizer for embedding model
        device: Device to use for embedding model
        dim: Dimension of embedding model output vector
        embedding_model_type: Type of embedding model
        """
        self.embedding_model = embedding_model
        self.dim = dim

    @abstractmethod
    def get_embeddings(self, texts):
        """
        gets embeddings for a list of texts
        must return embeddings as a torch tensor of shape (dim, len(texts))
        """
        pass

    def clear(self):
        # Free up memory
        # outputs = outputs.last_hidden_state.cpu()
        # token_embeddings = token_embeddings.cpu()
        # inputs = inputs.to('cpu')
        # del all_vertex_embeddings, vertex_embeddings, token_embeddings, inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, embedding_model, dim=384):
        super().__init__(embedding_model, dim)

    def get_embeddings(self, texts):
        """
        gets embeddings for a list of texts
        returns embeddings as a torch tensor of shape (dim, len(texts))
        """
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
        embeddings = embeddings.cpu().T
        # print("embeddings",embeddings)
        return embeddings


class TransformersEmbeddingModel(EmbeddingModel):
    def __init__(self, embedding_model, tokenizer, device="cuda", dim=768):
        """
        Intializes Transformer style Embedding model
        embedding_model: Embedding model
        tokenizer: Tokenizer for embedding model
        device: Device to use for embedding model
        embedding_dim: Dimension of embedding model
        """
        super().__init__(embedding_model, dim)
        self.tokenizer = tokenizer
        self.device = device

    def get_embeddings(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            token_embeddings = (
                outputs.last_hidden_state
            )  # Shape: (batch_size, seq_len, hidden_dim)
            embeddings = token_embeddings[:, 0, :].T  # Shape: (hidden_dim, batch_size)
            embeddings = embeddings.cpu()
        return embeddings


class GraphEmbedder:
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initializes the AnswerPathExtractor class.
        embedding_model: Embedding model
        embedding_model_type: Type of embedding model
        """
        self.embedding_model = embedding_model
        self.graph = None

    def add_embeddings_to_graph(self, graph, chunk_size=None):
        """
        Adds embeddings to a graph
        """
        if chunk_size is None:
            chunk_size = max(graph.num_vertices(), graph.num_edges())

        vertex_labels = []
        processed_nodes = 0
        all_vertex_embeddings = torch.empty(self.embedding_model.dim, 0)
        for index, vt in enumerate(graph.vertices()):
            if not graph.vertex_properties["terminal"][vt]:
                vertex_labels.append(graph.vertex_properties["label"][vt])
            else:
                vt_neighbors = vt.all_neighbors()
                first_adjacent_node = next(vt_neighbors)
                vertex_labels.append(
                    graph.vertex_properties["label"][first_adjacent_node]
                )

            if len(vertex_labels) >= chunk_size or (
                index >= graph.num_vertices() - 1 and len(vertex_labels) > 0
            ):  # add to graph
                vertex_embeddings = self.embedding_model.get_embeddings(vertex_labels)
                # print("vertex_embeddings", vertex_embeddings)
                all_vertex_embeddings = torch.hstack(
                    (all_vertex_embeddings, vertex_embeddings)
                )
                vertex_labels = []

        graph.vertex_properties["embedding"] = graph.new_vertex_property(
            "vector<float>"
        )
        graph.vertex_properties["embedding"].set_2d_array(all_vertex_embeddings)

        # Free up memory
        del all_vertex_embeddings, vertex_embeddings
        self.embedding_model.clear()

        # now do edges
        edge_labels = []
        processed_nodes = 0
        all_edge_embeddings = torch.empty(self.embedding_model.dim, 0)
        for index, e in enumerate(graph.edges()):
            edge_labels.append(graph.edge_properties["sentence"][e])

            if (
                len(edge_labels) >= chunk_size or index >= graph.num_edges() - 1
            ):  # add to graph
                edge_embeddings = self.embedding_model.get_embeddings(edge_labels)
                all_edge_embeddings = torch.hstack(
                    (all_edge_embeddings, edge_embeddings)
                )
                edge_labels = []

        graph.edge_properties["embedding"] = graph.new_edge_property("vector<float>")
        graph.edge_properties["embedding"].set_2d_array(all_edge_embeddings)

        # Free up memory
        del all_edge_embeddings, edge_embeddings
        self.embedding_model.clear()

    def add_embeddings_to_graphs(self, graphs, chunk_size=None):
        """
        Adds embeddings to multiple graphs
        """
        num_vertices = sum([g.num_vertices() for g in graphs])
        num_edges = sum([g.num_edges() for g in graphs])
        if chunk_size is None:
            chunk_size = max(num_vertices, num_edges)

        vertex_labels = []
        num_processed = 0
        all_vertex_embeddings = torch.empty(self.embedding_model.dim, 0)
        for g_index, graph in enumerate(graphs):
            for v_index, vt in enumerate(graph.vertices()):
                if not graph.vertex_properties["terminal"][vt]:
                    vertex_labels.append(graph.vertex_properties["label"][vt])
                else:
                    vt_neighbors = vt.all_neighbors()
                    first_adjacent_node = next(vt_neighbors)
                    vertex_labels.append(
                        graph.vertex_properties["label"][first_adjacent_node]
                    )
                num_processed += 1

                if len(vertex_labels) >= chunk_size or (
                    num_processed >= num_vertices - 1 and len(vertex_labels) > 0
                ):  # add to graph
                    vertex_embeddings = self.embedding_model.get_embeddings(
                        vertex_labels
                    )
                    # print("vertex_embeddings", vertex_embeddings)
                    all_vertex_embeddings = torch.hstack(
                        (all_vertex_embeddings, vertex_embeddings)
                    )
                    vertex_labels = []
        cur_index = 0
        for graph in graphs:
            end_index = cur_index + graph.num_vertices()
            graph.vertex_properties["embedding"] = graph.new_vertex_property(
                "vector<float>"
            )
            graph.vertex_properties["embedding"].set_2d_array(
                all_vertex_embeddings[:, cur_index:end_index]
            )
            cur_index = end_index

        # Free up memory
        del all_vertex_embeddings, vertex_embeddings
        self.embedding_model.clear()

        # now do edges
        edge_labels = []
        num_processed = 0
        all_edge_embeddings = torch.empty(self.embedding_model.dim, 0)
        for g_index, graph in enumerate(graphs):
            for e_index, e in enumerate(graph.edges()):
                edge_labels.append(graph.edge_properties["sentence"][e])
                num_processed += 1

                if len(edge_labels) >= chunk_size or (
                    num_processed >= num_edges - 1 and len(edge_labels) > 0
                ):  # add to graph
                    edge_embeddings = self.embedding_model.get_embeddings(edge_labels)
                    all_edge_embeddings = torch.hstack(
                        (all_edge_embeddings, edge_embeddings)
                    )
                    edge_labels = []

        cur_index = 0
        for graph in graphs:
            end_index = cur_index + graph.num_edges()
            graph.edge_properties["embedding"] = graph.new_edge_property(
                "vector<float>"
            )
            graph.edge_properties["embedding"].set_2d_array(
                all_edge_embeddings[:, cur_index:end_index]
            )
            cur_index = end_index

        # Free up memory
        del all_edge_embeddings, edge_embeddings
        self.embedding_model.clear()


class QueryAnswerTransformer:
    """
    Takes a query answers pair and extracts embeddings and entities from them
    """

    def __init__(self, nlp: Language, embedding_model: EmbeddingModel):
        self.nlp = nlp
        self.embedding_model = embedding_model

    def extract_metadata(self, qas: List[Tuple[str, List[str]]]):
        flattened_queries = [pair[0] for pair in qas]
        flattened_answers = [answer for pair in qas for answer in pair[1]]

        all_texts = flattened_queries + flattened_answers
        all_texts_docs = list(self.nlp.pipe(all_texts))

        all_query_ents = []
        for doc in all_texts_docs[: len(flattened_queries)]:
            ents = [ent.text for ent in doc.ents]
            all_query_ents.append(ents)

        all_answer_ents = []
        for doc in all_texts_docs[len(flattened_queries) :]:
            ents = [ent.text for ent in doc.ents]
            all_answer_ents.append(ents)

        query_embeddings = self.embedding_model.get_embeddings(flattened_queries).T

        result = []
        start = 0
        for q_index, pair in enumerate(qas):
            query = pair[0]
            answers = pair[1]
            query_entities = list(set(all_query_ents.pop(0)))

            # query_embeddings = query_embeddings.T
            # print("query_embeddings", query_embeddings.shape)
            query_embedding = query_embeddings[start : start + 1, :]  # shape (num_queries, embedding_dim)
            start += 1

            answer_entities = []
            for answer_option in pair[1]:
                answer_entities.extend(all_answer_ents.pop(0))

            answer_entities = list(set(answer_entities))
            result.append(
                {
                    "query": query,
                    "query_embedding": query_embedding,
                    "query_entities": query_entities,
                    "answers": answers,
                    "answer_entities": answer_entities,
                }
            )

        return result
