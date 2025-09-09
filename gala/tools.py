from llama_index.core import Document
import pandas as pd
from llama_index.retrievers.bm25 import BM25Retriever
from huggingface_hub import list_models
import random

def get_data(data) -> Document:
    guest_dataset = pd.read_csv(data)
    docs = [
        Document(
            text="\n".join([
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}"
            ]),
            metadata={"name": guest_dataset['name'][i]}
        )
        for i in range(len(guest_dataset))
    ]
    return docs

nodes = get_data("/Users/anantmishra/projects/huggingface_tutorials/gala/resources/display-first-10-train-rows.csv")
bm25_retriever = BM25Retriever.from_defaults(nodes =nodes)

def get_guest_info_retriever(query: str) -> str:
    """Retrieves guest information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([f"{doc.text}" for doc in results])
    else:
        return "No relevant information found about the guest."
    
def get_weather_info(location:str) -> str:
    """Fetches dummy weather information for a given location"""

    weather_condition = [
        {"condition":"Rainy", "temp_c":15},
        {"condition":"Clear", "temp_c":25},
        {"condition":"Windy", "temp_c":20},
    ]

    data = random.choice(weather_condition)
    return f"Weather in {location} : {data['condition']}, {data['temp_c']}"

def get_hub_stats(author:str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list(list_models(author = author, sort = "downloads", direction = -1, limit = 1))

        if models:
            model = model[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads"
        else:
            return f"No model found for this author {author}"
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"