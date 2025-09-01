import os
import chromadb
import asyncio
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow,ReActAgent
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import QueryEngineTool,FunctionTool

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_name="BAAI/bge-small-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=model_name, token=hf_token)

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection(name="attention_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store,embed_model = embed_model)

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
    provider="auto",
    streaming = False
)

query_engine = index.as_query_engine(llm=llm, response_mode="tree_summarize")

query_tool = QueryEngineTool.from_defaults(query_engine=query_engine,name="QueryEngine",description="Useful for when you need to answer questions about the context or if you need to look up information")

query_agent = ReActAgent(
    tools=[query_tool],
    llm=llm,
    system_prompt=(
        "You are a helpful research assistant. "
        "You have access to a tool that can help you look up information. "
        "Use the tool whenever you need to find information or answer questions about the context."
    ),
    name="QueryAgent",
    description="Agent that can answer questions about the context using a tool."
)


# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

calculator_agent = ReActAgent(
    name = "CalculatorAgent",
    llm = llm,
    system_prompt = "You are a helpful math assistant. You can perform basic arithmetic operations like addition and subtraction.",
    tools = [
        FunctionTool.from_defaults(add, name="add", description="Add two numbers"),
        FunctionTool.from_defaults(subtract, name="subtract", description="Subtract two numbers")
    ],
    description = "Agent that can perform basic arithmetic operations like addition and subtraction."
)

multi_agent = AgentWorkflow(
    agents=[
        query_agent,
        calculator_agent
    ],
    root_agent="CalculatorAgent",
)

async def run_query():
    query = "what is attention in deep learning models? what is three added to five? what is eighteen minus six?"
    result = await multi_agent.run(query)
    print(result)

asyncio.run(run_query())
