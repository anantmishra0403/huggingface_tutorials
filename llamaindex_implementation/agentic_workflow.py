import asyncio
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event
import random

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        print("Hello, World!")
        return StopEvent(result="Workflow completed")

class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        print("Step One")
        if random.random() < 0.5:
            return LoopEvent(loop_output="Looping back to Step One")
        return ProcessingEvent(intermediate_result="Processed in Step One")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        print("Step Two with:", ev.intermediate_result)
        return StopEvent(result="Multi-step Workflow completed")
    
async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    result = await w.run()
    print("Result:", result)
    mw = MultiStepWorkflow(timeout=10, verbose=False)
    result = await mw.run()
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(main())

