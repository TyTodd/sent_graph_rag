# SentGraph Rag
## ToDo
- test out using Avro instead of parquet

## Datasets
### SentenceGraphDataset
Representation of a dataset as graphs with embeddings
#### `__init__(in_path)`
Loads a graph datatset
#### `fromJson(in_path, out_path)`
Converts a json context-qas dataset to a graph dataset
```python3
gd = GraphDataset.fromJson("articles.json", "articlesGD.parquet")
```
**In format** (jsonl file)
```json
[
    {
    "context": "string",
    "qas" :[
        {
            "question": "string",
            "answers": ["string"],
        }
    ]
    }

]
```
**Out format** (parquet file)
```json
[
    {
        "context": "base64_encoded_graph_string",
        "qas": [
            {
                "question": "string",
                "question_embedding": [0.123, 0.456, ...],  // Fixed size 125
                "answers": ["str"]
            }
        ]
    }
]
```

### AnswerPathDataset
#### `__init__(in_path)`
Loads a graph datatset
#### `fromGraphDS(in_path, out_path)`
Converts a graph dataset to a answer path dataset used for model training

**In format** (parquet file)
```json
[
    {
        "context": "base64_encoded_graph_string",
        "qas": [
            {
                "question": "string",
                "question_embedding": [0.123, 0.456, ...],  // Fixed size 125
                "answers": ["str"]
            }
        ]
    }
]
```

**Out format** (parquet file)
```json
[
    {
    "path": [Vector[float32,125]],
    "options" :[Vector[float32,125]],
    "label": Int
    }
]
```


## Models
### TransformerExplorer
### LSTMExplorer

## Installation
Create Conda Environment (Required for graph-tool)
```
conda create --name sent_graph_rag -c conda-forge graph-tool python=3.11.11
conda activate sent_graph_rag

```
Install sent_graph_rag
```
pip install -e Desktop/sent_graph_rag
```
Install Spacy NLP model
```
python -m spacy download en_core_web_sm
```
# Engaging Usage Notes
```
srun -N 1 -n 1 --pty /bin/bash
```

# Satori Usage Notes
Start interactive session
```
srun --gres=gpu:1 -N 1 --mem=100G  --time 12:00:00  --pty /bin/bash
```
Allocates 1 GPU, 100GB of memory, and 12 hours of time.