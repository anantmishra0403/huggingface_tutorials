import os
from huggingface_hub import login
from dotenv import load_dotenv
from smolagents import Tool, CodeAgent,InferenceClientModel

load_dotenv()

login(token = os.getenv("HF_TOKEN"), add_to_git_credential = False)

class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = "This tool suggests creative superhero-themed party ideas based on a category. It returns a unique party theme idea."
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham')"
        }
    }
    output_type = "string"

    def forward(self, category: str) -> str:
        themes = {
            "classic heroes": "A party where everyone dresses as their favorite classic superhero, with themed decorations and activities.",
            "villain masquerade": "A mysterious masquerade ball where guests come as iconic villains, complete with dark decor and themed cocktails.",
            "futuristic Gotham": "A high-tech party set in a futuristic version of Gotham City, featuring neon lights and advanced gadgets."
        }
        return themes.get(category.lower(), "Theme party not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")
    
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(
    model=InferenceClientModel(),
    tools=[party_theme_tool]
)

result = agent.run(""" 
    What would be a good superhero-themed party idea for a 'villain masquerade'? theme
    """)
print(result)