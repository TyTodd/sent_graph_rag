import io
import graph_tool as gt

def print_json_schema(data, indent=0):
    """Recursively print the schema of a JSON object."""
    if isinstance(data, dict):
        print(" " * indent + "{")
        for key, value in data.items():
            print(" " * (indent + 2) + f'"{key}": {type(value).__name__}', end="")
            if isinstance(value, (dict, list)):
                print(" ->")
                print_json_schema(value, indent + 4)
            else:
                print(",")
        print(" " * indent + "}")
    elif isinstance(data, list):
        print(" " * indent + "[")
        if data:
            print_json_schema(data[0], indent + 2)
        else:
            print(" " * (indent + 2) + "Empty list")
        print(" " * indent + "]")
    else:
        print(" " * indent + f"{type(data).__name__}")

def string_to_graph(graph_string):
    # Use BytesIO to read the binary data
    buffer = io.BytesIO(graph_string)
    graph = gt.Graph()  # Create an empty graph
    graph.load(buffer)  # Load the graph from the buffer
    return graph

def graph_to_string(graph):
    # Use BytesIO for binary data
    buffer = io.BytesIO()
    graph.save(buffer)  # Save in default .gt format
    return buffer.getvalue()  # Return the binary data