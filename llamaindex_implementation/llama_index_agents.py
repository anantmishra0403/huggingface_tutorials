from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
import asyncio


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=hf_token,
    provider="auto",
    streaming = False
    )

agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply, name="multiply", description="Multiply two numbers")],
    llm=llm
)

async def run_query(context: Context, query: str):
    result = await agent.run(query, context=context)
    print(f"Q: {query}\nA: {result}\n")


async def main():
    ctx = Context(agent)
    await run_query(None, "What is 6 multiplied by 7?")
    await run_query(ctx, "My name is Anant Mishra")
    await run_query(ctx, "What is my name again?")

# Kick off
if __name__ == "__main__":
    asyncio.run(main())

