from sent_graph_rag.Datasets.answer_path_dataset import extract_answer_paths_from_entities

vertex_embeddings = {
    "Sofie": [0.21380899, 0.92684816, 0.30860686],
    "MIT": [0.13363062, 0.98783793, -0.0794926],
    "NYC": [0.16035675, 0.98635373, 0.03731008],
    "1861": [0.02672612, 0.88133184, -0.47174135],
    "Cambridge": [-0.05345225, 0.70833497, -0.70384972]
}

edge_embeddings = {
    "1": [0.16035675, 0.98635373, 0.03731008],
    "2": [0.21380899, 0.92684816, 0.30860686],
    "3": [-0.1069045, 0.5487777, -0.82910462],
    "4": [-0.35540099, 0.42761799, 0.83116364],
    "5": [0.05345225, 0.92214396, -0.38313623],
    "6": [0.05345225, 0.92214396, -0.38313623],
    "7": [0.05345225, 0.92214396, -0.38313623],
    "8": [0.1069045, 0.97639568, -0.18767764],
    "9": [-0.05345225, 0.70833497, -0.70384972]
}

@pytest.mark.parametrize("question_entities, answer_entities, answers, expected_num_paths", [
    pytest.param(["New York"], [],["Cambridge"], 2, False, id="source_entity_match"), # 1 of question_entities is in least similar vertex aliases, 4
    #question entity empty, 3
    pytest.param([], [],["Cambridge"], 1, False, id="target_exact_match"), # answer should be an alias of target
    pytest.param([], ["Cambridge"],["is in Cambridge"], 1, False, id="target_edge_entity_match"), # answer should be in final edge sentence -- connected to target u should go through, answer_entity that matches the target's aliases 
    pytest.param([], [], ["is in Cambridge"], 1, True, id="target_edge_match"), # same as last but answer entity is blank -- targetIsEdge=True
    pytest.param([], ["Cambridge"],["Cambridge, Massachusetts"], 1, False, id="target_entity_match"), # answer entities in target aliases, answer is defined but not in edge (random string)
    pytest.param([], [],["Cambridge, Massachusetts"], 1, False, id="no_path"), # answer random strings
])

def test_answer_path_dataset(question_entities, answer_entities, answers, expected_num_paths, targetIsEdge):
    '''
    What it do
    1. Finds nodes that are in the graph that are similar to the question
        Testing functions directly by making a test graph
        Defining our own embeddings (3D vectors)
    2. 
    
    (1) Make a graph (6 verticies)
        1-ideal source node (most related to question, has to have the same entities/aliases as the question)
        1-ideal target node (most related to answer; exact match, edge entitiy match, edge match, entity match)
        4-other nodes
        2 paths from source to target
        should return shortest path

    (2) Run from answer path dataset -- extract answer paths from entities 
        Takes in graph, question, question eneitirs, embeddings, answer + answer entities 
        2 questions
        Have an expected path that should be returned 

        question_entities -- should be 1 of the aliases from the source node, in another test it shouldnt be
        each edge sentence should be different
            target edge in 1 case should contain an answer entity (no answer)
            in another should have answer but not answer entity
            another case - within nodes aliases contains just the answer
            case - within the target node the answer is not in aliases but one of the answer's entities are in the aliases

    '''
    # Graph Creation
    graph = IGraphSentenceGraph("test corpus")
    question = "What city does Sofie go to school in?"
    question_entities

    # Vertices
    soruce = graph.add_vertex({ # [0.21380899, 0.92684816, 0.30860686]
        "id": "1",
        "label": "", #doesnt matter
        "terminal": False, #always 
        "ner_label": "", #doesnt matter
        "aliases": ["Sofie", "Sofie's"] #matters
    })
    v2 = graph.add_vertex({ #[0.13363062, 0.98783793 -0.0794926]
        "id": "2",
        "label": "",
        "terminal": False,
        "ner_label": "",
        "aliases": ["MIT", "Massachusetts Institute of Technology"]
    })
    v3 = graph.add_vertex({ # [0.16035675, 0.98635373, 0.03731008]
        "id": "3",
        "label": "",
        "terminal": False,
        "ner_label": "",
        "aliases": ["New York"]
    })
    v4 = graph.add_vertex({ # [ 0.02672612, 0.88133184, -0.47174135]
        "id": "4",
        "label": "",
        "terminal": False,
        "ner_label": "",
        "aliases": ["1861"]
    })
    target = graph.add_vertex({ # [-0.05345225, 0.70833497, -0.70384972]
        "id": "5",
        "label": "",
        "terminal": False,
        "ner_label": "",
        "aliases": ["Cambridge"]
    })
    
    # Edges
    edge_prop = {
        "sentence": "Sofie is currently taking NLP at MIT.",
        "entity1": "Sofie",
        "entity2": "MIT",
        "terminal": False
    }
    edge1 = graph.add_edge(source, v2, edge_prop) #weight: 2, embedding: [0.16035675, 0.98635373, 0.03731008]
    
    edge_prop = {
        "sentence": "Sofie wants to visit New York, but she needs to work on her NLP project.",
        "entity1": "Sofie",
        "entity2": "New York",
        "terminal": False
    }
    edge2 = graph.add_edge(source, v3, edge_prop) #weight: 1, embedding: [0.21380899, 0.92684816, 0.30860686]
    
    edge_prop = {
        "sentence": "MIT, also known as the Massachusetts Institute of Technology, is in Cambridge",
        "entity1": "MIT",
        "entity2": "Cambridge",
        "terminal": False
    }
    edge3 = graph.add_edge(v2, target, edge_prop) #weight: 7, embedding: [-0.1069045, 0.5487777, -0.82910462]
    
    edge_prop = {
        "sentence": "MIT was founded in 1861.",
        "entity1": "MIT",
        "entity2": "1861",
        "terminal": False
    }
    edge4 = graph.add_edge(v2, v4, edge_prop) #weight: 1, embedding: [-0.35540099, 0.42761799, 0.83116364]

    edge_prop = {
        "sentence": "In 1861, Sofie's great-grandparents moved to New York.",
        "entity1": "1861",
        "entity2": "Sofie's",
        "terminal": False
    }
    edge5 = graph.add_edge(v4, source, edge_prop) #weight: 4, embedding: [0.05345225, 0.92214396, -0.38313623]
    
    edge_prop = {
        "sentence": "In 1861, Sofie's great-grandparents moved to New York.",
        "entity1": "1861",
        "entity2": "New York",
        "terminal": False
    }
    edge6 = graph.add_edge(v4, v3, edge_prop) #weight: 4, embedding: [0.05345225, 0.92214396, -0.38313623]
    
    edge_prop = {
        "sentence": "In 1861, Sofie's great-grandparents moved to New York.",
        "entity1": "Sofie's",
        "entity2": "New York",
        "terminal": False
    }
    edge7 = graph.add_edge(source, v3, edge_prop) #weight: 4, embedding: [0.05345225, 0.92214396, -0.38313623]

    edge_prop = {
        "sentence": "1861 was a great year for Cambridge.",
        "entity1": "1861",
        "entity2": "Cambridge",
        "terminal": False
    }
    edge8 = graph.add_edge(v4, target, edge_prop) #weight: 3, embedding: [0.1069045, 0.97639568, -0.18767764]

    edge_prop = {
        "sentence": "Sofie went to visit MIT while we was in high school.",
        "entity1": "Sofie",
        "entity2": "MIT",
        "terminal": False
    }
    edge9 = graph.add_edge(source, v2, edge_prop) #weight: 6, embedding: [-0.05345225, 0.70833497, -0.70384972]

    # Edge Embeddings
    graph.add_edge_embeddings(torch.tensor([edge_embeddings["1"],
                                            edge_embeddings["2"],
                                            edge_embeddings["3"],
                                            edge_embeddings["4"],
                                            edge_embeddings["5"],
                                            edge_embeddings["6"],
                                            edge_embeddings["7"],
                                            edge_embeddings["8"],
                                            edge_embeddings["9"]
                                            ]))

    # Vertex Embeddings
    graph.add_vertex_embeddings(torch.tensor([vertex_embeddings["Sofie"],
                                            vertex_embeddings["MIT"],
                                            vertex_embeddings["NYC"]
                                            vertex_embeddings["1861"]
                                            vertex_embeddings["Cambridge"]
                                            ]))

    # Question Embedding
    question_embedding = [1.0, 2.0, 3.0]

    # Correct Path
    correct_path1 = [
                        [torch.tensor([vertex_embeddings["Sofie"]])],
                        [
                            torch.tensor([edge_embeddings["1"]]),
                            torch.tensor([edge_embeddings["2"]]),
                            torch.tensor([edge_embeddings["5"]]),
                            torch.tensor([edge_embeddings["7"]]),
                            torch.tensor([edge_embeddings["9"]])
                        ],
                        [torch.tensor([vertex_embeddings["MIT"]])],
                        [
                            torch.tensor([edge_embeddings["4"]]),
                            torch.tensor([edge_embeddings["3"]]),
                            torch.tensor([edge_embeddings["1"]]),
                            torch.tensor([edge_embeddings["9"]])
                        ],
                        [torch.tensor([vertex_embeddings["1861"]])],
                        [
                            torch.tensor([edge_embeddings["8"]]),
                            torch.tensor([edge_embeddings["4"]]),
                            torch.tensor([edge_embeddings["5"]]),
                            torch.tensor([edge_embeddings["6"]])
                        ],
                        [torch.tensor([vertex_embeddings["Cambridge"]])]
                    ]
    if targetIsEdge:
        correct_path1.pop(-1)

    correct_path2 = [
                        [torch.tensor([vertex_embeddings["NYC"]])],
                        [
                            torch.tensor([edge_embeddings["6"]]),
                            torch.tensor([edge_embeddings["2"]]),
                            torch.tensor([edge_embeddings["7"]])
                        ],
                        [
                            torch.tensor([vertex_embeddings["1861"]]),
                            torch.tensor([vertex_embeddings["Sofie"]]),
                            torch.tensor([vertex_embeddings["NYC"]])
                        ],
                        [
                            torch.tensor([edge_embeddings["8"]]),
                            torch.tensor([edge_embeddings["4"]]),
                            torch.tensor([edge_embeddings["5"]]),
                            torch.tensor([edge_embeddings["6"]])
                        ],
                        [torch.tensor([vertex_embeddings["Cambridge"]])]
                    ]
    
    correct_paths = [correct_path1, correct_path2]


    data = extract_answer_paths_from_entities(graph, [question], [question_entities], [question_embedding], [answer_entiies], [answers])
    result_question_embedding, paths = data[0]
    assert result_question_embedding == question_embedding, "question embedding doesn't match"
    assert len(paths) ==  expected_num_paths, "unexpected number of paths"
    #check if path is correct
    for i, path in enumerate(paths):
        correct_path = correct_paths[i]
        assert len(path) == len(correct_path), "lens of path {i} don't match"
        for j, options in enumerate(path):
            correct_options = correct_path[j]
            assert len(options) == len(correct_options), "path {i}, lens of option {j} don't match"
            assert torch.isclose(options[0], correct_options[0]), "first option doesn't match --> for path {i}, option {j} isn't close enough"
            for option in options:
                assert any(torch.isclose(t, option) for t in correct_options), "for path {i}, option {j}, isn't close enough"