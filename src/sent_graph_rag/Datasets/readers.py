from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TypedDict, Iterator
import json

class QAS(TypedDict):
    question: str
    answers: List[str]

class Row(TypedDict):
    context: str
    qas: List[QAS] 



class DatasetReader(ABC):
    """
    Abstract base class for dataset readers.
    Defines the interface for reading and processing datasets.
    """
    @abstractmethod
    def __init__(self, file_path: str):
        """
        Initializes the dataset reader.
        """
        pass
    
    @abstractmethod
    def read(self) -> Iterator[Row]:
        """
        Reads a dataset from a file and yields rows.
        
        Args:
            file_path: Path to the dataset file
            chunk_size: Number of rows to yield at a time
        Yields:
            Rows from the dataset (context and list of questions and answers as a dictionary)
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of rows in the dataset.
        """
        pass

class SQUADReader(DatasetReader):
    """
    Reader for the SQuAD dataset.
    """
    def __init__(self, file_path: str):
        super().__init__(file_path)
        with open(file_path, 'r') as f:
            self.squad_data = json.load(f)
        self.data_length = sum([sum([1 for i in topic['paragraphs']]) for topic in self.squad_data['data']])
        
    def read(self) -> Iterator[Row]:
        for topic in self.squad_data["data"]:
            for paragraph in topic["paragraphs"]:
                yield Row(context=paragraph["context"], qas=paragraph["qas"])
    
    def __len__(self) -> int:
        return self.data_length
        
