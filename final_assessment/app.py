import os
import gradio as gr
import requests
import inspect
import pandas as pd
import asyncio
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult, AgentStream
from llama_index.core.llms.callbacks import llm_completion_callback
from agent_workflow import get_agent
import os
import gradio as gr
import requests
import inspect
import pandas as pd
import shutil
from io import BytesIO
from PIL import Image
import pandas as pd
import json
import io

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
class BasicAgent:
    def __init__(self):
        self.workflow = get_agent()
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

async def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        files_url = f"{api_url}/files/{task_id}"
        print(files_url)
        file_type = None
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            files_response = requests.get(files_url)
            print(files_response)
            content_type = files_response.headers.get("Content-Type", "")
            print("Content-Type:", content_type)
            if files_response.status_code == 200:
                if "image" in content_type:
                    # Handle image
                    file_type = 'image'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.png"
                    with open(file_path, "wb") as f:
                        f.write(files_response.content)

                elif "audio" in content_type or "mpeg" in content_type:
                    # Handle audio
                    file_type = 'audio'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.mp3"
                    with open(file_path, "wb") as f:
                        f.write(files_response.content)
                    print("Saved an audio file.")

                elif "csv" in content_type or "text/csv" in content_type:
                    # Handle CSV
                    file_type = 'csv'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.csv"
                    df = pd.read_csv(BytesIO(response.content))
                    df.to_csv(file_path, index=False)
                    print("CSV loaded into DataFrame:")
                    print(df.head())

                elif "json" in content_type:
                    file_type = 'json'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.json"
                    text = BytesIO(files_response.content).getvalue().decode("utf-8")
                    parsed = json.loads(text)
                    with open(file_path, "w") as f:
                        json.dump(parsed, f)
                    print("✅ Saved JSON as output.json")

                elif "application/octet-stream" in content_type:
                    file_type = 'excel'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.xlsx"
                    df = pd.read_excel(BytesIO(files_response.content))
                    df.to_excel(file_path, index=False)

                elif "python" in content_type:
                    file_type = 'python'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.py"
                    with open(file_path,"wb") as f:
                        f.write(files_response.content)

                else:
                    # Fallback: just save raw file
                    file_type = 'file'
                    file_path = f"/Users/anantmishra/projects/Final_Assignment_Template/files/{task_id}.bin"
                    with open(file_path, "wb") as f:
                        f.write(files_response.content)
                    print("Saved unknown file type as output.bin")
            else:
                print("Failed to fetch file:", response.status_code)
            if file_type:
                question_text = f"{question_text} {file_type} path : {file_path}"
            print(question_text)
            submitted_answer = await agent(question_text)
            print(submitted_answer)
            precise_answer = ""
            if "FINAL ANSWER" in submitted_answer:
                precise_answer = submitted_answer.split("FINAL ANSWER: ")[1]
            elif "Final Answer" in submitted_answer:
                precise_answer = submitted_answer.split("Final Answer: ")[1]
            elif "Final answer" in submitted_answer:
                precise_answer = submitted_answer.split("Final answer: ")[1]
            elif "final answer" in submitted_answer:
                precise_answer = submitted_answer.split("final answer: ")[1]
            elif "final_answer" in submitted_answer:
                precise_answer = submitted_answer.split("final_answer: ")[1]
            
            
            
            if "[" in precise_answer and "]" in precise_answer:
                if "," not in precise_answer:
                    precise_answer = precise_answer.replace("[","").replace("]","")

            if precise_answer == "":
                precise_answer = submitted_answer

            print(precise_answer)
            submitted_answer = precise_answer
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)