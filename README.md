# SentGraph Rag
## ToDo
- test out using Avro instead of parquet

## Datasets
### GraphDataset
Representation of a dataset as graphs with embeddings
#### fromJson(in_path, out_path)
```python3
GraphDataset.fromJson("articles.json", "articlesGD.parquet")
```
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

#### AnswerPathDataset

## Models
### TransformerExplorer
### LSTMExplorer
