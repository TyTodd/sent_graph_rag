import time
from spacy.tokens import Doc

def create_graph_from_doc(GraphClass, doc: Doc, verbose: bool = False):
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
    alias_eid_map = {} #maps each alias (reference that is a PROPN) to the entity id that it is an alias of
    unique_entities = {} # maps each unique entity id to a list of its references across the text
    eid_name_map = {} # mapping entity ids to the name of the entiy
    eid_label_map = {} # mapping entity ids to the label of the entity
    for ent in doc.ents:
            # unique id will be {entityText}_{label}
            ent_loc = (ent.start_char, ent.end_char)
            if ent_loc not in visited_references:
                if ent.text in alias_eid_map and eid_label_map[alias_eid_map[ent.text]] == ent.label_:
                    entity_id = alias_eid_map[ent.text]
                else:
                    entity_id = f"{ent.text}_{ent.label_}"
                    eid_name_map[entity_id] = ent.text
                    eid_label_map[entity_id] = ent.label_

                unique_entities.setdefault(entity_id, [])

                if ent_loc in entity_cluster_map: # if in a cluster map to all references of entity
                    visited_references.update(set(entity_cluster_map[ent_loc]))
                    unique_entities[entity_id].extend(entity_cluster_map[ent_loc])
                    for loc in entity_cluster_map[ent_loc]:
                        ref = doc.char_span(loc[0], loc[1], alignment_mode="contract")
                        pos = [tok.pos_ for tok in ref]
                        if "PROPN" in pos:
                            alias_eid_map[ref.text] = entity_id
              
              
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

    graph = data_to_graph(GraphClass, eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = verbose)

    if verbose: print(f"Graph Build Time: {time.time() - graph_build_start}")
    return graph


def data_to_graph(GraphClass, eid_name_map, unique_entities,  doc, reference_eid_map, eid_label_map, sentence_ref_map, eid_alias_map, verbose = False):
    id_to_vertex = {}
    graph = GraphClass(doc.text)

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
    graph.id_to_vertex = id_to_vertex
    return graph