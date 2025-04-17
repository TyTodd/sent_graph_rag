import spacy
from spacy.language import Language
import torch
import gc
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union, Optional
import numpy as np
from numpy.linalg import norm
import time
from .graph import SentenceGraph
from .sentence_graph_dataset import SentenceGraphDataset
import os
import lmdb
from pathlib import Path
import pickle
from torch.utils.data import Dataset

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

def shortest_path_to_edge(graph: SentenceGraph, start_node, target_edge, weights= None):
    source_vertex = target_edge.source()
    target_vertex = target_edge.target()

    path_source, path_target = graph.shortest_paths(start_node, [source_vertex, target_vertex])
    shortest_path_source, edge_list_source = path_source
    shortest_path_target, edge_list_target = path_target
    
    path, edge_list = None, None
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


def shortest_path_to_edge_group(graph: SentenceGraph, start_node, edge_group):
  shortest = None
  nodes = set()
  
  edge_sentence = graph.get_edge_property(edge_group[0], "sentence") #graph.edge_properties['sentence'][edge_group[0]]
  for edge in edge_group:
    source, target = graph.edge_endpoints(edge)
    nodes.add(source)
    nodes.add(target)

  # handle edge case when start_node is on edge
  if start_node in nodes:
    return [start_node], [edge_group[0]]

  shortest = None
  shortest_length = float('inf')
  paths = graph.shortest_paths(start_node, nodes)
  
  for path in paths:
    path_length = graph.path_length(path)
    if (path_length < shortest_length) and len(path[0]) != 0:
      shortest_length = path_length
      shortest = path
      
  # NOTE: This code was in a working version but should cause an error
  # if len(path) == 0:
  #   # print("NO PATH")
  #   return [], []
  # print("SHORTEST", shortest)
  
  # Our path needs to end in an edge instead of a node
  # So if the last edge is in the edge group. Great! we can remove the last node
  # and that will be our ending edge
  # However, if the last edge is not in the edge group that means we found the shortest path to 
  # one of the nodes connected to and edge in the edge group without going through the edge
  # So we just append *any* edge in the edge group (but we use the first one since they all 
  # have the same sentence/embedding) to the end of the path
  path, edge_list = shortest
  if graph.get_edge_property(edge_list[-1], "sentence") == edge_sentence:
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
  remapped_similarities = 1 - ((cosine_similarities + 1) / 2) # remap to [0, 1] 0 is most similar 1 is least similar
  
  return list(remapped_similarities)
  # edge_weights = graph.new_edge_property("float")

  # edge_weights.set_values(list(remapped_similarities))
  # return edge_weights


class AnswerPathDataset(Dataset):
  def __init__(self, path_name: str, verbose: bool = False):
    """
    Initializes the AnswerPathExtractor class.
    nlp: Spacy nlp model
    tokenizer: Tokenizer for embedding model
    embedding_model: Embedding model
    device: Device to use for embedding model
    embedding_dim: Dimension of embedding model
    embedding_model_type: Type of embedding model
    """
    self.path_name = path_name
    self.verbose = verbose
    self.env = lmdb.open(path_name, readonly=True, lock=False)
    with self.env.begin() as txn:
      self.length = txn.stat()["entries"]
    
    def __len__(self):
      return self.length

    def __getitem__(self, idx):
      with self.env.begin() as txn:
        raw = txn.get(str(idx).encode())
        sample = pickle.loads(raw)
        return sample
    
    

  @classmethod
  def from_graph_dataset(self, graph_dataset: Union[SentenceGraphDataset, str], out_path: Union[str, Path], verbose: bool = False, k: int = 3):
    if isinstance(graph_dataset, str):
      graph_dataset = SentenceGraphDataset.from_graph_dataset(graph_dataset, verbose)
    embedding_dim = graph_dataset.metadata["embedding_dim"]
    row_size = 41 * embedding_dim * 4 # max of 41 vectors per row (1 query, 20 MAX components in path, 20 MAX options) (float32)
    num_paths = graph_dataset.data_length * 10 * 10 #assuming 10 questions per graph and 10 paths per question # TODO: count number of questions
    num_rows = num_paths * 20 # each path expands into path_length rows (20 is max path length)
    dataset_size = int(row_size * num_rows * 2) # safety factor of 2
    out_path = Path(out_path)
    out_path = out_path.with_suffix('.lmdb')
    
    env = lmdb.open(out_path, map_size=dataset_size)
    with env.begin(write=True) as txn:
      for i, sample in enumerate(get_answer_path_rows(graph_dataset,k)):
        txn.put(str(i).encode(), pickle.dumps(sample))
        
    answer_path_dataset = AnswerPathDataset(out_path, verbose)
    
    return answer_path_dataset
    
    
  
def get_answer_path_rows(graph_dataset: SentenceGraphDataset, k: int = 3):
  for record in graph_dataset:
    graph = record["graph"]
    all_queries = record["queries"]
    all_query_entities = record["query_entities"]
    all_query_embeddings = record["query_embeddings"]
    all_answer_entities = record["answer_entities"]
    all_answers = record["answers"]
    answer_paths = extract_answer_paths_from_entities(graph, all_queries, all_query_entities, all_query_embeddings, all_answer_entities, all_answers, k)
    expanded_answer_paths = expand_answer_path_data(answer_paths)
    for row in expanded_answer_paths:
      yield row
      
      
      

def get_source_nodes(graph: SentenceGraph, query_entities, query_embedding, k = 3):
  def matches_entity(aliases):
    for alias in aliases:
      if alias in query_entities:
        return True
    return False
  query_embedding = np.array(query_embedding).reshape(1, 384)

  graph.set_vertex_filter("aliases", filter_fn = matches_entity)
  entity_matches = [v for v in graph.vertices()]
  graph.clear_filters()
  query_embedding = torch.from_numpy(query_embedding)

  graph.set_vertex_filter("terminal", eq_value=False) # TODO: make sure this is correct.  
  # this is how it was done before with graph-toolgraph.set_vertex_filter(graph.vertex_properties["terminal"], inverted=True)
  # embedding_prop = graph.vertex_properties["embedding"]
  # embeddings = embedding_prop.get_2d_array().T
  graph.get_vertex_embeddings()
  query_embedding = query_embedding.double()

  embeddings = torch.from_numpy(embeddings)
  query_norm = torch.norm(query_embedding)

  embeddings_norms = torch.norm(embeddings, dim=1)

  #query_embedding shape (1, dim_size)
  #embedding shape (num_vertices, dim_size)
  scores = (embeddings @ query_embedding.T) / (embeddings_norms[:, None] * query_norm)
  verticies = list(graph.iter_vertices())

  k = min(k, len(scores))
  values, indices = torch.topk(scores, k=k, dim=0, largest=True)

  embedding_matches = []
  for index in indices:
    embedding_matches.append(verticies[index])

  graph.clear_filters()
  return set(embedding_matches + entity_matches)


def get_target_nodes(answer_entities: List[str], answers: List[str], graph: SentenceGraph, matching = "edge_match"):
  """
  Get the target nodes from a graph that matche the answers to a query
  answer_entiies: The NER entities extracted from the answers
  answers: the original answers
  graph: gt.Graph to search in
  matching: The matching method to use
    exact_match - only looks for nodes whose aliases are an exact match for the answer
    edge_entity_match - everything above but if exact_match fails returns nodes whose aliases match the answer entities connected to edges that contain the answer
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

  graph.set_vertex_filter("aliases", filter_fn = exact_match)
  exact_matches = [v for v in graph.iter_vertices()]
  graph.clear_filters()
  if len(exact_matches) > 0 or matching == "exact_match":
    return exact_matches, False
    # pass #TODO do not forget to remove

  # If no exact matches were found and exact_match = False see if a sentence matches
  graph.set_edge_filter("sentence", filter_fn = edge_match, filter_unconnected_vertices=True)
  matched_edges = [e for e in graph.iter_edges()]
  #Now that edges not containing the answer are filtered out find filter out nodes that are no longer connected
  # (We do this step with filter_unconnected_vertices=True)

  #Now get all nodes which have entity matches
  graph.set_vertex_filter("aliases", filter_fn = entity_match)
  sentence_entity_matches = [v for v in graph.iter_vertices()]
  graph.clear_filters()
  if len(sentence_entity_matches) > 0 or matching == "edge_entity_match":
    return sentence_entity_matches, False

  if len(matched_edges) > 0 or matching == "edge_match":
    edge_groups = {}
    for edge in matched_edges:
      sent = graph.get_edge_property(edge, 'sentence')
      edge_groups.setdefault(sent, [])
      edge_groups[sent].append(edge)
    return list(edge_groups.values()), True

  # If we still have no target nodes do entity match
  graph.set_vertex_filter("aliases", filter_fn = entity_match)
  entity_matches = [v for v in graph.iter_vertices()]
  graph.clear_filters()
  return entity_matches, False

def extract_answer_paths_from_entities(graph: SentenceGraph, all_queries: List[str], all_query_entities: List[List[str]], 
                                        all_query_embeddings: List[np.ndarray], all_answer_entities: List[List[str]], 
                                        all_answers: List[List[str]], k: int = 3) -> List[Tuple[np.ndarray, List[List[List[np.ndarray]]]]]:
  """
  Extract answer paths from entities and embeddings
  Returns:
    all_data: list of tuples of (query_embedding, embedding_data)
      embedding_data: list of paths
        path: list of options_lists
          options_list: list of embedding options
  """
  # extract entities and embeddings from query
  all_data = []
  for q_index, query_entities in enumerate(all_query_entities):
    query = all_queries[q_index]
    query_embedding = all_query_embeddings[q_index]
    answer_entities = all_answer_entities[q_index]
    answers = all_answers[q_index]

    # start = time.time()
    source_nodes = get_source_nodes(graph, query_entities, query_embedding, k = k)
    # print("Found Source nodes", time.time() - start)
    # start = time.time()
    targets, targets_are_edges = get_target_nodes(answer_entities, answers, graph, matching = "edge_match")
    # print("Found target nodes", time.time() - start)

    if len(source_nodes) == 0 or len(targets) == 0:
      statement = "No source nodes found" if len(source_nodes) == 0 else "No target nodes found"
      # print(statement)
      continue
    remap_cos_sim = lambda x: 1-((x+1)/2)
    #add weights to adges
    # start = time.time()
    edge_weights = get_similarity_weights(graph, query_embedding)
    graph.set_edge_weights(edge_weights)
    # print("Calculated weights", time.time() - start)
    paths = []
    counter = 0
    # start = time.time()
    # TODO: change to shortest_paths implementation
    for source in source_nodes:
      if targets_are_edges:
        for target in targets:
          path = shortest_path_to_edge_group(graph, source, target)
          # print("Found Shortest path to edge group", time.time() - sub_start)
          if len(path[0]) != 0:
            paths.append(path)
      else:
        # sub_start = time.time()
        paths = graph.shortest_paths(source, targets)
        # print("Found Shortest path to node", time.time() - start)
        paths.extend([path for path in paths if len(path[0]) != 0])
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
        # same_edge = lambda e: graph.edge_properties['sentence'][e] == graph.edge_properties['sentence'][edge]
        edge_group_sentence = graph.get_edge_property(edge, 'sentence')
        source_node = path[i]

        if i == 0:
          path_data.append([source_node])
          embedding = list(graph.get_vertex_property(source_node, 'embedding'))
          path_embedding_data.append([embedding])

        # add all edge options for correct node
        incorrect_edges = []
        added_sentences = set()
        # start = time.time()
        for edge_option in graph.get_edges(source_node):
          if graph.get_edge_property(edge_option,'sentence') != graph.get_edge_property(edge, 'sentence') and graph.get_edge_property(edge_option, 'sentence') not in added_sentences:
            incorrect_edges.append(edge_option)
            added_sentences.add(graph.get_edge_property(edge_option, 'sentence'))
        row = [edge] + incorrect_edges
        path_data.append(row)
        path_embedding_data.append([list(graph.get_edge_property(e, 'embedding')) for e in row])

        # print("added edge options:", time.time() - start)
        # add all node options for correct edge
        start = time.time()
        if i < len(path) - 1:
          target_node = path[i+1]
          # TODO: change in filter method here for graph-tool
          graph.set_edge_filter("sentence", eq_value = edge_group_sentence) 
          # print("set sentence match filter:", time.time() - start)
          added_nodes = set([target_node])
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
          path_embedding_data.append([list(graph.get_vertex_property(n, 'embedding')) for n in row])
          graph.clear_filters()
        # print("single iteration time:" , time.time() - iter_start)

      # get the last step of a path and add one more row for training the stop vector
      last_step = path_data[-1][0] #get the last correct edge or vertex in the path
      if graph.is_edge(last_step):
        last_node = path_data[-2][0]
        graph.set_edge_filter("sentence", eq_value = graph.get_edge_property(last_step, 'sentence'))
        added_nodes = set([last_node])
        incorrect_nodes = []
        for e in graph.edges():
          for n in graph.edge_endpoints(e):
            if n not in added_nodes:
              incorrect_nodes.append(n)
              added_nodes.add(n)
        graph.clear_filters()
        row = [None] + incorrect_nodes
        path_data.append(row)
        path_embedding_data.append([None] + [list(graph.get_vertex_property(n, 'embedding')) for n in incorrect_nodes])
      else:
        incorrect_edges = []
        added_sentences = set()
        for edge_option in graph.get_edges(last_step):
          if graph.get_edge_property(edge_option, 'sentence') not in added_sentences:
            incorrect_edges.append(edge_option)
            added_sentences.add(graph.get_edge_property(edge_option, 'sentence'))
        row = [None] + incorrect_edges
        path_data.append(row)
        path_embedding_data.append([None] + [list(graph.get_edge_property(e, 'embedding')) for e in incorrect_edges])

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
  
def expand_answer_path_data(data: List[Tuple[np.ndarray, List[List[List[np.ndarray]]]]]) -> List[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]]:
  """
  Expand the answer path data to include all possible paths
  Params:
    data: list of tuples of (query_embedding, embedding_data)
      embedding_data: list of paths
        path: list of options_lists
          options_list: list of embedding options
  Returns:
    list of training_rows
      training_row: (query_embedding, path_so_far, options) 
        query_embedding: embedding of the query
        path_so_far: list of embeddings of the path so far
        options: list of options for the next step (first option is always correct)
  """
  expanded_data = []
  for query_embedding, embedding_data in data:
    for path in embedding_data:
      path_so_far = [path[0][0]] # first list in the options list always has a single vertex. (start node)
      for options in path[1:]:
        expanded_data.append((query_embedding, path_so_far, options))
        path_so_far.append(options[0])
  return expanded_data

def print_path(graph , path, edge_list):
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



