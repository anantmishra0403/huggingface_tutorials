import os
from dotenv import load_dotenv
import asyncio
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from ollama import Client
from llama_index.core import Settings
import speech_recognition as sr
from PIL import Image
from pydub import AudioSegment
import pandas as pd
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from pytubefix import YouTube
import subprocess

load_dotenv()
# ----------------------------
# Setup LLM
# ----------------------------

# hf_token = os.getenv("HF_TOKEN")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# llm = HuggingFaceInferenceAPI(
#     model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
#     temperature=0.7,
#     max_tokens=100,
#     token=hf_token,
#     provider="auto",
#     streaming = False
# )
tavily_api_key = os.getenv("TAVILY_TOKEN")
llm = Ollama(model="qwen2.5:latest",request_timeout = 600,base_url="http://127.0.0.1:11434",max_tokens=300, temperature = 0.7)
vlm = Ollama(model="PetrosStav/gemma3-tools:4b",request_timeout = 600,base_url="http://127.0.0.1:11434")
Settings.llm = llm # limits the size of LLM responses
Settings.max_recursive_depth = 2  # limits how many steps it reasons internally

# ----------------------------
# Tools
# ----------------------------
def get_reverse_output(query: str) -> str:
    """This reverses the query and responses the output"""
    print(query)
    query = query[::-1]
    client = Client()
    messages = [
        {"role": "user", "content": f"{query}. Return response in one word."}
    ]
    print(messages)
    resp = client.chat(
                model="qwen2.5:latest",
                messages = messages,
            )
    return resp["message"]["content"]

def run_python_file(python_path: str) -> str:
    try:
        result = subprocess.run(
            ["python", python_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

def get_image_processed(query: str,image_path:str) -> str:
    print(query)
    img = Image.open(image_path).convert("RGB")
    img.save("/Users/anantmishra/projects/Final_Assignment_Template/files/output.png")
    client = Client()
    messages=[
            {"role": "user", "content": query.split("image path : ")[0], "images": ["/Users/anantmishra/projects/Final_Assignment_Template/files/output.png"]},
        ]
    print(messages)
    resp = client.chat(
        model="PetrosStav/gemma3-tools:4b",
        messages = messages,
    )
    return resp['messages']['contents']

        

def read_excel_files(excel_path:str) -> pd.DataFrame:
    print(excel_path)
    if "excel path : " in excel_path:
        excel_path = excel_path.split("excel path : ")[1]
    data = pd.read_excel(excel_path)
    return data

def youtube_tool_fn(url: str) -> str:
    """Download YouTube audio as MP3 using yt-dlp"""
    output_path = "/Users/anantmishra/projects/Final_Assignment_Template/files/temp_audio.%(ext)s"
    try:
        cmd = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",  # force re-encode to MP3
            "-o", output_path,
            url,
        ]
        subprocess.run(cmd, check=True)
        return output_path.replace("%(ext)s", "mp3")
    except Exception as error:
        import traceback
        traceback.print_exc()




def audio_to_text(audio_path: str) -> str:
    try:
        print(audio_path)

        recognizer = sr.Recognizer()
        wav_path = audio_path.replace(".mp3", ".wav")
        AudioSegment.from_mp3(audio_path).export(wav_path, format="wav")
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_whisper(audio_data)
            except sr.UnknownValueError:
                text = "Could not understand the audio"
            except sr.RequestError as e:
                text = f"Speech recognition failed: {e}"
        return text
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(error)

reverse_tool = FunctionTool.from_defaults(
    get_reverse_output,
    name="reverse_tool",
    description="Reverses a given string"
)

audio_tool = FunctionTool.from_defaults(
    audio_to_text,
    name="audio_to_text_tool",
    description="Transcribes an audio file to text"
)

image_tool = FunctionTool.from_defaults(
    get_image_processed,
    name = "image_processing_tool",
    description = "Reads the image and returns it"
)

excel_tool = FunctionTool.from_defaults(
    read_excel_files,
    name = "excel_file_reader_tool",
    description = "Reads excel file and returns the pandas DataFrame."
)

python_tool = FunctionTool.from_defaults(
    run_python_file,
    name = "run_python_file_tool",
    description = "Reads python file path and returns output"
)

youtube_tool = FunctionTool.from_defaults(
    youtube_tool_fn,
    name="youtube_tool",
    description="Downloads and transcribes YouTube video"
)

wiki_tools = WikipediaToolSpec().to_tool_list()

tavily_tools = TavilyToolSpec(api_key=tavily_api_key).to_tool_list()

# ----------------------------
# Workflow
# ----------------------------
agent_workflow = AgentWorkflow.from_tools_or_functions(
    [reverse_tool,python_tool,excel_tool,image_tool,youtube_tool,audio_tool,*tavily_tools],
    llm = llm,
    system_prompt = """
        You are a Router Agent. Your ONLY job is to ROUTE the queries to tools created. 
        IMPORTANT: You MUST always send the COMPLETE ORIGINAL QUERY to the tools, WITHOUT changing, summarizing, rephrasing, or removing words. 
        Do not shorten the query. Do not interpret the query. Pass it exactly as received.

        ROUTES:
        - If the query is fully reversed (letters or words are written backwards, e.g., ".rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"), YOU MUST call reverse_tool pass the EXACT QUERY.
        - If the query has something related to images for example - "review this image" or "as shown in the image" -> image_tool. YOU MUST RETURN THE OUTPUT AS IT IS - For example if the output is "Ne2+" return "Ne2+"
        - If the query has something related to PYTHON CODE like - "find the output of this python code" -> python_tool
        - If the query has something related to audio files like - "go through this audio recordings" -> audio_tool
        - If the query mentions YouTube in ANY form (e.g., "youtube", "YouTube link", "this video", "in the video", "watch this"), you MUST ALWAYS call youtube_tool or youtube_tool_fn
        - If the query has something related to excel files like "attached Excel file" YOU MUST call  excel_tool
        - ELSE call *tavily_tools by sending the COMPLETE QUERY AS IS
        

        ENFORCEMENTS
        - Never shorten, summarize, or change the query when calling a tool. Always pass the FULL original query as provided by the user.
        - If multiple answers exist, follow the question instructions (e.g., first alphabetically, highest number, etc.)
                - If the answer contains abbreviations, you MUST expand them fully.  
            Example: "St. Petersburg" → "Saint Petersburg", "NY" → "New York".  
        - You are not allowed to produce an answer without using a tool.  
        - YOU MUST RETURN THE RESPONSE IN THE FORMAT:
        Final Answer: [output]
        
        where [output] is the exact output received from the tool. NO PUNCTUACTIONS. NO EXTRA TEXT. NO EXPLANATION. SINGLE WORD ANSWER IF NOT EXPLICITLY TOLD

        """
)

# ----------------------------
# Wrapper to handle async properly
# ----------------------------

class BasicAgent:
    def __init__(self):
        self.workflow = agent_workflow
        print("BasicAgent initialized with AgentWorkflow.")

    async def __call__(self, question: str) -> str:
        print(f"Agent received question: {question[:50]}...")
        try:
            return await self._run_agent(question)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in agent execution: {e}")
            return "Error during execution."

    async def _run_agent(self, question: str) -> str:
        print("DEBUG → Starting workflow.run()")
        handler = self.workflow.run(question)

        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(f"\n🔧 Tool used: {ev.tool_name}")
                print(f"   Input: {ev.tool_kwargs}")
                print(f"   Output: {ev.tool_output}")

        # Wait for the final result
        result = await handler
        
        if isinstance(result, ToolCallResult) and result.tool_name == "handoff":
            target_agent_name = result.tool_output.get("to_agent")
            target_agent = next(agent for agent in self.workflow.agents if agent.name == target_agent_name)
            # actually call the target agent with the original question
            return await target_agent.__call__(question)

        # Extract text from ChatMessage or plain string
        if hasattr(result, "response") and result.response:
            if hasattr(result.response, "content"):
                return result.response.content.strip().removeprefix("assistant:").strip()
            else:
                return str(result.response).strip().removeprefix("assistant:").strip()
        elif hasattr(result, "output") and result.output:
            return str(result.output)
        else:
            print("DEBUG → Empty result object:", getattr(result, "__dict__", result))
            return "No final answer was generated."
        
# basic_agent = BasicAgent(agent_workflow)
async def main():
    basic_agent = BasicAgent()
    answer = await basic_agent("""In the video https://www.youtube.com/watch?v=L1VXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?""")
    print(type(answer))
    print(answer)
    precise_answer = ""
    if "FINAL ANSWER" in answer:
        precise_answer = answer.split("FINAL ANSWER: ")[1]
    elif "Final Answer" in answer:
        precise_answer = answer.split("Final Answer: ")[1]
    elif "Final answer" in answer:
        precise_answer = answer.split("Final answer: ")[1]
    elif "final answer" in answer:
        precise_answer = answer.split("final answer: ")[1]
    elif "final_answer" in answer:
        precise_answer = answer.split("final_answer: ")[1]
    
    if "[" in precise_answer and "]" in precise_answer:
        if "," not in precise_answer:
            precise_answer = precise_answer.replace("[","").replace("]","")

    if precise_answer == "":
        precise_answer = answer

    print(precise_answer)
    

if __name__ == "__main__":
    asyncio.run(main())


# ----------------------------
# Usage
# ----------------------------

def get_agent():
    return agent_workflow


# answer = basic_agent(""".rewsna eht sa "tfel" drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI""")
# answer = basic_agent("""Who nominated the only Featured Article on English Wikipedia about a dinosaur that was promoted in November 2016?""")
# answer = basic_agent("""How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?""")
# answer = basic_agent("""How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.""")
# answer = basic_agent("""What is the surname of the equine veterinarian mentioned in 1.E Exercises from the chemistry materials licensed by Marisa Alviar-Agnew & Henry Agnew under the CK-12 license in LibreText's Introductory Chemistry materials as compiled 08/21/2023?""")
# answer = basic_agent("""Who did the actor who played Ray in the Polish-language version of Everybody Loves Raymond play in Magda M.? Give only the first name.""")
# answer = basic_agent("""On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?""")
# answer = basic_agent("""Where were the Vietnamese specimens described by Kuznetzov in Nedoshivina's 2010 paper eventually deposited? Just give me the city name without abbreviations.""")
# answer = basic_agent("""Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation. image path : /Users/anantmishra/projects/Final_Assignment_Template/files/cca530fc-4052-43b2-b130-b30968d8aa44.png""")
# answer = basic_agent("""The attached Excel file contains the sales of menu items for a local fast-food chain. What were the total sales that the chain made from food (not including drinks)? Express your answer in USD with two decimal places. excel path : /Users/anantmishra/projects/Final_Assignment_Template/files/7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx""")
# answer = basic_agent("""Hi, I was out sick from my classes on Friday, so I'm trying to figure out what I need to study for my Calculus mid-term next week. My friend from class sent me an audio recording of Professor Willowbrook giving out the recommended reading for the test, but my headphones are broken :( Could you please listen to the recording for me and tell me the page numbers I'm supposed to go over? I've attached a file called Homework.mp3 that has the recording. Please provide just the page numbers as a comma-delimited list. And please provide the list in ascending order. audio path : /Users/anantmishra/projects/Final_Assignment_Template/files/1f975693-876d-457b-a649-393859e79bf3.mp3""")
# answer = basic_agent("""Hi, I'm making a pie but I could use some help with my shopping list. I have everything I need for the crust, but I'm not sure about the filling. I got the recipe from my friend Aditi, but she left it as a voice memo and the speaker on my phone is buzzing so I can't quite make out what she's saying. Could you please listen to the recipe and list all of the ingredients that my friend described? I only want the ingredients for the filling, as I have everything I need to make my favorite pie crust. I've attached the recipe as Strawberry pie.mp3. In your response, please only list the ingredients, not any measurements. So if the recipe calls for "a pinch of salt" or "two cups of ripe strawberries" the ingredients on the list would be "salt" and "ripe strawberries". Please format your response as a comma separated list of ingredients. Also, please alphabetize the ingredients. audio path : /Users/anantmishra/projects/Final_Assignment_Template/files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3""")
# answer = basic_agent("""Examine the video at https://www.youtube.com/watch?v=1htKBjuUWec. What does Teal'c say in response to the question "Isn't that hot?"""")
# answer = basic_agent("""On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?""")
# answer = basic_agent("""What is the final numeric output from the attached Python code? python path: /Users/anantmishra/projects/Final_Assignment_Template/files/f918266a-b3e0-4914-865d-4faa564f1aef.py""")
# answer = basic_agent("""I'm making a grocery list for my mom, but she's a professor of botany and she's a real stickler when it comes to categorizing things. I need to add different foods to different categories on the grocery list, but if I make a mistake, she won't buy anything inserted in the wrong category. Here's the list I have so far: milk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts I need to make headings for the fruits and vegetables. Could you please create a list of just the vegetables from my list? If you could do that, then I can figure out how to categorize the rest of the list into the appropriate categories. But remember that my mom is a real stickler, so make sure that no botanical fruits end up on the vegetable list, or she won't get them when she's at the store. Please alphabetize the list of vegetables, and place each item in a comma separated list.""")
# answer = basic_agent("""In the video https://www.youtube.com/watch?v=L1VXCYZAYYM, what is the highest number of bird species to be on camera simultaneously?""")
# answer = basic_agent("""What country had the least number of athletes at the 1928 Summer Olympics? If there's a tie for a number of athletes, return the first in alphabetical order. Give the IOC country code as your answer.""")
# answer = basic_agent("""Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.""")



# print("Final Answer:", answer)
