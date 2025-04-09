import spacy
from spacy.language import Language
import torch
import gc
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from numpy.linalg import norm
from graph_tool.all import shortest_path
import time

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (norm(a) * norm(b))

def list_structure(lst):
    if isinstance(lst, list):
        # Handle lists
        if not lst:  # Empty list
            return "List[Any]"
        inner_types = {list_structure(item) for item in lst}
        if len(inner_types) == 1:
            return f"List[{inner_types.pop()}]"
        else:
            return f"List[Union[{', '.join(inner_types)}]]"
    elif isinstance(lst, tuple):
        # Handle tuples
        if not lst:  # Empty tuple
            return "Tuple[Any, ...]"
        inner_types = [list_structure(item) for item in lst]
        return f"Tuple[{', '.join(inner_types)}]"
    elif isinstance(lst, str):
        return "str"
    elif isinstance(lst, int):
        return "int"
    elif isinstance(lst, float):
        return "float"
    elif isinstance(lst, dict):
        return "Dict"
    elif lst is None:
        return "None"
    else:
        return type(lst).__name__

def shortest_path_to_edge(graph, start_node, target_edge, weights= None):
    source_vertex = target_edge.source()
    target_vertex = target_edge.target()

    path, edge_list = shortest_path(graph, start_node, source_vertex, weights = weights)

    shortest_path_source, edge_list_source = shortest_path(graph, start_node, source_vertex, weights = weights)
    shortest_path_target, edge_list_target = shortest_path(graph, start_node, target_vertex, weights = weights)
    if len(edge_list_source) < len(edge_list_target):
      path, edge_list = shortest_path_source, edge_list_source
    else:
      path, edge_list = shortest_path_target, edge_list_target

    if len(path) == 0 or len(edge_list) == 0:
      return [], []
    #if its in the path already remove the last node other wise add it
    if edge_list[-1] == target_edge:
      path.pop(-1)
    else:
      edge_list.append(target_edge)
      #TODO: doesnt check the edge case if you skip the edge in the shortest path from start_node to source or target.
      #(parallel eddge between source and target on the shortest path from start to either)

    return path, edge_list


def shortest_path_to_edge_group(graph, start_node, edge_group, weights = None):
  shortest = None
  nodes = set()
  edge_sentence = graph.edge_properties['sentence'][edge_group[0]]
  for edge in edge_group:
    nodes.add(edge.source())
    nodes.add(edge.target())

  # handle edge case when start_node is on edge
  if start_node in nodes:
    return [start_node], [edge_group[0]]

  shortest = None
  for node in nodes:
    path, edge_list = shortest_path(graph, start_node, node, weights = weights)
    if (shortest is None or len(path) < len(shortest[0])) and len(path) != 0:
      shortest = (path, edge_list)

  if len(path) == 0:
    # print("NO PATH")
    return [], []
  # print("SHORTEST", shortest)
  path, edge_list = shortest
  if graph.edge_properties['sentence'][edge_list[-1]] == edge_sentence:
    path.pop(-1)
  else:
    edge_list.append(edge_group[0])

  return shortest

# def get_similarity_weights(graph, query_embedding):
#   remap_cos_sim = lambda x: 1-((x+1)/2)
#   similarity_weight = lambda x: remap_cos_sim(cosine_similarity(x, query_embedding))
#   edge_weights = graph.new_edge_property("float")
#   weights = graph.edge_properties["embedding"].transform(similarity_weight, value_type="float")
#   return weights

def get_similarity_weights(graph, query_embedding):
  remap_cos_sim = lambda x: 1-((x+1)/2)
  similarity_weight = lambda x: remap_cos_sim(cosine_similarity(x, query_embedding))
  edge_embeddings = graph.edge_properties["embedding"].get_2d_array()

  query_embedding = np.array(query_embedding).reshape(1,384) #shape [1, 384]
  edge_embeddings = edge_embeddings.T  # Shape: [num_edges, 384]

  # Step 2: Normalize query and edge_embeddings
  query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)  # Shape: [1, 384]
  edge_embeddings_norm = edge_embeddings / np.linalg.norm(edge_embeddings, axis=1, keepdims=True)  # Shape: [num_edges, 384]

  # Step 3: Compute cosine similarities
  cosine_similarities = np.dot(edge_embeddings_norm, query_norm.T).flatten()  # Shape: [num_edges]
  remapped_similarities = (cosine_similarities + 1) / 2
  edge_weights = graph.new_edge_property("float")

  edge_weights.set_values(list(remapped_similarities))
  return edge_weights


class AnswerPathExtractor:
  def __init__(self, nlp: Language = None, embedding_model: EmbeddingModel = None):
    """
    Initializes the AnswerPathExtractor class.
    nlp: Spacy nlp model
    tokenizer: Tokenizer for embedding model
    embedding_model: Embedding model
    device: Device to use for embedding model
    embedding_dim: Dimension of embedding model
    embedding_model_type: Type of embedding model
    """
    self.nlp = nlp
    self.embedding_model = embedding_model
    self.graph = None

  def set_graph(self, graph):
    self.graph = graph

  def get_source_nodes(self, graph, query_entities, query_embedding, k = 3):
    def matches_entity(aliases):
      for alias in aliases:
        if alias in query_entities:
          return True
      return False
    query_embedding = np.array(query_embedding).reshape(1, 384)

    alias_prop = graph.vertex_properties["aliases"]
    filter_prop = alias_prop.transform(matches_entity, value_type="bool")
    graph.set_vertex_filter(filter_prop)
    entity_matches = [v for v in graph.vertices()]
    graph.clear_filters()
    query_embedding = torch.from_numpy(query_embedding)

    graph.set_vertex_filter(graph.vertex_properties["terminal"], inverted=True)
    embedding_prop = graph.vertex_properties["embedding"]
    embeddings = embedding_prop.get_2d_array().T
    query_embedding = query_embedding.double()

    embeddings = torch.from_numpy(embeddings)
    query_norm = torch.norm(query_embedding)

    embeddings_norms = torch.norm(embeddings, dim=1)

    #query_embedding shape (1, dim_size)
    #embedding shape (num_vertices, dim_size)
    scores = (embeddings @ query_embedding.T) / (embeddings_norms[:, None] * query_norm)
    verticies = list(graph.vertices())

    k = min(k, len(scores))
    values, indices = torch.topk(scores, k=k, dim=0, largest=True)

    embedding_matches = []
    for index in indices:
      embedding_matches.append(verticies[index])

    graph.clear_filters()
    return set(embedding_matches + entity_matches)


  def get_target_nodes(self, answer_entities: List[str], answers: List[str], graph: gt.Graph, matching = "edge_match"):
    """
    Get the target nodes from a graph that matche the answers to a query
    answer_entiies: The NER entities extracted from the answers
    answers: the original answers
    graph: gt.Graph to search in
    matching: The matching method to use
      exact_match - only looks for nodes whose aliases are an exact match for the answer
      edge_entity_match - everything above but if exact_match fails returns nodes whose entities match the answer connected to edges that contain the answer
      edge_match - everything above but if edge_entity_match fails returns EDGES whose sentences contain the answer
      entity_match - everything above but if edge_match fails returns nodes whose aliases match the answer entities
    """

    #Returns true if an answer matches an alias of a node (most accurate and best for training objective)
    def exact_match(aliases):
      for alias in aliases:
        if alias in answers: #String in list
          return True
      return False

    #Returns true if an answer is inside and edge sentence (also very accurare but not the best for training objective)
    #After finding edges need to determine which node to traverse to by seeing if the entities match
    def edge_match(sentence):
      for answer in answers:
        if answer in sentence: # string in string
          return True
      return False

    #Returns true if an answer entity matches any alias of a node (Least accurate)
    def entity_match(aliases):
      for alias in aliases:
        if alias in answer_entities: #string in list
          return True
      return False

    alias_prop = graph.vertex_properties["aliases"]
    exact_match_prop = alias_prop.transform(exact_match, value_type="bool")
    graph.set_vertex_filter(exact_match_prop)
    exact_matches = [v for v in graph.vertices()]
    graph.clear_filters()
    if len(exact_matches) > 0 or matching == "exact_match":
      return exact_matches, False
      # pass #TODO do not forget to remove

    # If no exact matches were found and exact_match = False see if a sentence matches
    edge_match_prop = graph.edge_properties["sentence"].transform(edge_match, value_type="bool")
    graph.set_edge_filter(edge_match_prop)

    matched_edges = [e for e in graph.edges()]
    #Now that edges not containing the answer are filtered out find filter out nodes that are no longer connected
    connected_filter = graph.new_vertex_property("bool")
    connected_filter.a = graph.get_out_degrees(graph.get_vertices()) > 0
    graph.set_vertex_filter(connected_filter)

    #Now get all nodes which have entity matches
    alias_prop = graph.vertex_properties["aliases"]
    entity_match_prop = alias_prop.transform(entity_match, value_type="bool")
    graph.set_vertex_filter(entity_match_prop)
    sentence_entity_matches = [v for v in graph.vertices()]
    graph.clear_filters()
    if len(sentence_entity_matches) > 0 or matching == "edge_entity_match":
      return sentence_entity_matches, False
      # pass # TODO remove this

    if len(matched_edges) > 0 or matching == "edge_match":
      edge_groups = {}
      for edge in matched_edges:
        sent = graph.edge_properties['sentence'][edge]
        edge_groups.setdefault(sent, [])
        edge_groups[sent].append(edge)
      return list(edge_groups.values()), True

    # If we still have no target nodes do entity match
    alias_prop = graph.vertex_properties["aliases"]
    entity_match_prop = alias_prop.transform(entity_match, value_type="bool")
    graph.set_vertex_filter(entity_match_prop)
    entity_matches = [v for v in graph.vertices()]

    graph.clear_filters()
    return entity_matches, False

  def batch_extract_answer_paths(self, graphs: List[gt.Graph], all_queries: List[List[str]], all_answers: List[List[List[str]]]):
    assert self.embedding_model != None, "Embedding model must be set to batch extract answer paths"
    assert self.nlp != None, "Nlp model must be set to batch extract answer paths"
    # extract entities and embeddings from query
    flattened_queries = [query for queries in all_queries for query in queries]
    # print("flattened_queries", len(flattened_queries))
    all_query_embeddings = self.embedding_model.get_embeddings(flattened_queries).T #shape (num_embeddings, dim_size)
    flattened_answers =  [
                            answer
                            for answers in all_answers
                            for answer_options in answers
                            for answer in answer_options
                        ]

    all_texts = flattened_queries + flattened_answers
    all_texts_docs = list(self.nlp.pipe(all_texts))

    all_query_ents = []
    for doc in all_texts_docs[:len(flattened_queries)]:
      ents = [ent.text for ent in doc.ents]
      all_query_ents.append(ents)

    all_answer_ents = []
    for doc in all_texts_docs[len(flattened_queries):]:
      ents = [ent.text for ent in doc.ents]
      all_answer_ents.append(ents)

    # extract entities and embeddings from answers

    #match entitities to
    start = 0
    all_data = []
    for g_index, graph in enumerate(graphs):
      num_queries = len(all_queries[g_index])
      query_entities = [list(set(ents)) for ents in all_query_ents[:num_queries]]
      del all_query_ents[:num_queries]

      query_embeddings = all_query_embeddings[start:start+num_queries,:] #shape (num_queries, embedding_dim)
      start = start + num_queries

      answer_entities = []
      for answer_options in all_answers[g_index]:
        num_answers = len(answer_options)
        answers =  all_answer_ents[:num_answers]
        flattened_answers = [answer for answer_entities in all_answer_ents[:num_answers] for answer in answer_entities]
        answer_entities.append(list(set(flattened_answers)))
        del all_answer_ents[:num_answers]
      query_embeddings = query_embeddings.numpy()
      query_embeddings = [list(query_embeddings[i:i+1, :].flatten()) for i in range(query_embeddings.shape[0])]
      data = self.extract_answer_paths_from_entities(graph, all_queries[g_index], query_entities, query_embeddings, answer_entities, all_answers[g_index])
      all_data.extend(data)
      # break #TODO DO NOT FORGET TO REMOVE
    return all_data

  def extract_answer_paths_from_entities(self, graph:gt.Graph, all_queries: List[str], all_query_entities: List[List[str]], all_query_embeddings: List[List[float]], all_answer_entities: List[List[str]], all_answers: List[List[str]]):
    # extract entities and embeddings from query
    all_data = []
    for q_index, query_entities in enumerate(all_query_entities):
      query = all_queries[q_index]
      query_embedding = all_query_embeddings[q_index]
      answer_entities = all_answer_entities[q_index]
      answers = all_answers[q_index]

      # start = time.time()
      source_nodes = self.get_source_nodes(graph, query_entities, query_embedding, k = 3)
      # print("Found Source nodes", time.time() - start)
      # start = time.time()
      targets, targets_are_edges = self.get_target_nodes(answer_entities, answers, graph, matching = "edge_match")
      # print("Found target nodes", time.time() - start)

      if len(source_nodes) == 0 or len(targets) == 0:
        statement = "No source nodes found" if len(source_nodes) == 0 else "No target nodes found"
        # print(statement)
        continue
      remap_cos_sim = lambda x: 1-((x+1)/2)
      #add weights to adges
      # start = time.time()
      edge_weights = get_similarity_weights(graph, query_embedding)
      # print("Calculated weights", time.time() - start)
      paths = []
      counter = 0
      # start = time.time()
      for source in source_nodes:
        for target in targets:
          if isinstance(target, list):
            # sub_start = time.time()
            path = shortest_path_to_edge_group(graph, source, target, weights = edge_weights)
            # print("Found Shortest path to edge group", time.time() - sub_start)
            if len(path[0]) != 0:
              paths.append(path)
          else:
            # sub_start = time.time()
            path = shortest_path(graph, source=source, target=target, weights = edge_weights)
            # print("Found Shortest path to node", time.time() - start)
            if len(path[0]) != 0:
              paths.append(path)
      # print("Found Shortest paths", time.time() - start)
      if len(paths) == 0:
        # graph.clear_filters()
        # print("No paths found")
        # print("QUERY:", query)
        # # print("SOURCE NODES:", source_nodes)
        # # print("TARGET NODES:", targets)
        # # print("Edges", list(graph.edges()))
        # print("SOURCE NODES:")
        # for node in source_nodes:
        #   print(graph.vertex_properties['label'][node] +":","v_index:", graph.vertex_index[node], "num_edges", list(node.all_edges()))
        #   # for edge in node.all_edges():
        #   #   print(graph.edge_properties['sentence'][edge])
        #   # print("-----------------------------------------")
        # print("TARGET NODES:")
        # for target in targets:
        #   if isinstance(target, list):
        #     print(graph.edge_properties['sentence'][target[0]])
        #   else:
        #     print(graph.vertex_properties['label'][target])
        #     print("All target edges")
        #     print("type", type(target))
        #     for edge in graph.edges():
        #       if edge.source() == target or edge.target() == target:
        #         print(graph.edge_properties['sentence'][edge])
        #   print("-----------------------------------------")

        continue
      data = [] # list of (query, List[path_data])
      embedding_data = []
      # loop_start = time.time()
      for path, edge_list in paths:
        path_data = [] #list of list of options [[v1a], [e1a, e1b, e1c], [v2a, v2b, v2c], [e3a, e3b, e3c], [v4a, v4b, v4c]] first option is always correct
        # [query1, [v1a], [e1a, e1b, e1c]],
        # [query1, [v1a, e1a], [v2a, v2b, v2c]],
        path_embedding_data = []
        for i, edge in enumerate(edge_list):
          # iter_start = time.time()
          same_edge = lambda e: graph.edge_properties['sentence'][e] == graph.edge_properties['sentence'][edge]
          same_sentence = lambda s: graph.edge_properties['sentence'][edge] == s
          source_node = path[i]

          if i == 0:
            path_data.append([source_node])
            embedding = list(graph.vertex_properties['embedding'][source_node])
            path_embedding_data.append([embedding])

          # add all edge options for correct node
          incorrect_edges = []
          added_sentences = set()
          # start = time.time()
          for edge_option in source_node.all_edges():
            if graph.edge_properties['sentence'][edge_option] != graph.edge_properties['sentence'][edge] and graph.edge_properties['sentence'][edge_option] not in added_sentences:
              incorrect_edges.append(edge_option)
              added_sentences.add(graph.edge_properties['sentence'][edge_option])
          row = [edge] + incorrect_edges
          path_data.append(row)
          path_embedding_data.append([list(graph.edge_properties['embedding'][e]) for e in row])

          # print("added edge options:", time.time() - start)
          # add all node options for correct edge
          start = time.time()
          if i < len(path) - 1:
            target_node = path[i+1]
            edge_match_prop = graph.edge_properties["sentence"].transform(same_sentence, value_type="bool")
            graph.set_edge_filter(edge_match_prop)
            # print("set sentence match filter:", time.time() - start)
            added_nodes = set([source_node, target_node])
            incorrect_nodes = []
            # start = time.time()
            for e in graph.edges():
              for n in (e.source(), e.target()):
                if n not in added_nodes:
                  incorrect_nodes.append(n)
                  added_nodes.add(n)
            # print("added all incorrect nodes to graph", time.time() - start)
            row = [target_node] + incorrect_nodes
            path_data.append(row)
            path_embedding_data.append([list(graph.vertex_properties['embedding'][n]) for n in row])
            graph.clear_filters()
          # print("single iteration time:" , time.time() - iter_start)

        # get the last step of a path and add one more row for training the stop vector
        last_step = path_data[-1][0] #get the last correct edge or vertex in the path
        if isinstance(last_step, gt.Edge):
          last_node = path_data[-2][0]
          same_sentence = lambda s: graph.edge_properties['sentence'][last_step] == s
          edge_match_prop = graph.edge_properties["sentence"].transform(same_sentence, value_type="bool")
          graph.set_edge_filter(edge_match_prop)
          added_nodes = set([last_node])
          incorrect_nodes = []
          for e in graph.edges():
            for n in (e.source(), e.target()):
              if n not in added_nodes:
                incorrect_nodes.append(n)
                added_nodes.add(n)
          graph.clear_filters()
          path_data.append(incorrect_nodes)
          path_embedding_data.append([list(graph.vertex_properties['embedding'][n]) for n in incorrect_nodes])
        else:
          incorrect_edges = []
          added_sentences = set()
          for edge_option in last_step.all_edges():
            if graph.edge_properties['sentence'][edge_option] not in added_sentences:
              incorrect_edges.append(edge_option)
              added_sentences.add(graph.edge_properties['sentence'][edge_option])
          path_data.append(incorrect_edges)
          path_embedding_data.append([list(graph.edge_properties['embedding'][e]) for e in incorrect_edges])

        data.append(path_data)
        embedding_data.append(path_embedding_data)

      # print first path
      # for row in data:
      #   print("PATH START")
      #   for i, options in enumerate(row):
      #     if i % 2 == 0:
      #       print("VERTEX OPTIONS")
      #       for v in options:
      #         print(graph.vertex_properties['label'][v])
      #         print("--------------------------------------------")
      #     else:
      #       print("EDGE OPTIONS")
      #       for e in options:
      #         print(graph.edge_properties['sentence'][e])
      #         print("--------------------------------------------")
      #   print("PATH END")

      # print("Converted paths to path data", time.time() - loop_start)
      all_data.append((query_embedding, embedding_data))
      # print("LENGTYHS",len(all_data), len(embedding_data))
      # print("LENGTYH inner start",len(all_data[0][1][0]), len(embedding_data[0][1][0]))
      # print("LENGTYH inner end",len(all_data[0][1][-1]), len(embedding_data[0][1][-1]))
    return all_data

  def print_path(self, graph , path, edge_list):
    if len(path) == 0:
      print("Empty Path")
      return
    print("Path")
    for i, edge in enumerate(edge_list):
      source = path[i]
      print("Source:",graph.vertex_properties['label'][source])
      print("Edge:", graph.edge_properties['sentence'][edge])
      if i < len(path) - 1:
        target = path[i+1]
        print("Target:",graph.vertex_properties['label'][target])
      print()
    print("End of path")

  def convert_to_answer_path_dataset(in_folder, out_path):
    pass


