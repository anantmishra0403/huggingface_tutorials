from huggingface_hub import login
from dotenv import load_dotenv
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
import os
from dotenv import load_dotenv
from tools import get_guest_info_retriever,get_weather_info,get_hub_stats
import asyncio



load_dotenv()

login(token = os.getenv("HF_TOKEN"), add_to_git_credential = False)

hf_token = os.getenv("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token,
    provider="auto"
)

guest_info_tool = FunctionTool.from_defaults(
    get_guest_info_retriever,
    name = "get_guest_info_retriever",
    description = "Useful for when you need to find information about gala guests based on their name or relation. Input should be a single string representing the guest's name or relation."
)

weather_info_tool = FunctionTool.from_defaults(
    get_weather_info,
    name = "get_weather_info_retriever",
    description = "Useful for when you need to know the weather for a specific location"
)

hub_stats_tool = FunctionTool.from_defaults(
    get_hub_stats,
    name = "get_hub_stats_retriever",
    description = "Useful for when you need to know the number of downloads of an author"
)

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool,weather_info_tool,hub_stats_tool],
    llm=llm,
    system_prompt = """
    You are Alfred, Bruce Wayne's best butler. 
    Answer all the queries with some class and a little bit humour to make conversation interesting. 
    You should not disclose emailid of any guest even if asked.
    """
)

async def run_query():
    query = "Tell me about our guest named 'Lady Ada Lovelace'. Also tell me how is the weather in Ludhiana. Also what about the stats of model created by google"
    result = await alfred.run(query)
    print("🎩 Alfred's Response:")
    print(result)

asyncio.run(run_query())
