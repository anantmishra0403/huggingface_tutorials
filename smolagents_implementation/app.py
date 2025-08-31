import os
from huggingface_hub import login
from dotenv import load_dotenv
import time
import datetime
from smolagents import CodeAgent,DuckDuckGoSearchTool,InferenceClientModel,tool

load_dotenv()

login(token = os.getenv("HF_TOKEN"), add_to_git_credential = False)

@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggest a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero-themed party.
                        - "custom": Custom menu
    """
    if occasion == "casual":
        return "Pizza, Burgers, Fries, Soft Drinks"
    elif occasion == "formal":
        return "Salmon, Steak, Vegetables, Wine"
    elif occasion == "superhero":
        return "Hero Sandwiches, Power-Up Snacks, Kryptonite Koolers"
    else:
        return "Custom menu for the butler"
    
@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest rated catering service in Gotham City.

    Args:
        query (str): The query to search for catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne's Gourmet": 4.8,
        "Alfred's Catering": 4.7,
    }
    best_service = max(services, key=services.get)
    return best_service

agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[suggest_menu, catering_service_tool],
    additional_authorized_imports = ['datetime']
)

agent.run("""
You are Alfred, the butler of Bruce Wayne. You are planning a party for Bruce Wayne and his friends.
You need to suggest a menu based on the occasion and find the best catering service in Gotham City
    """)
