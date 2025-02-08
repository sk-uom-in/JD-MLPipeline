from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool,TXTSearchTool
import os
from generate import *

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

directory = "/mnt/iusers01/fse-ugpgt01/compsci01/n66425sa/hack/JD-MLPipeline/task4/checklist.txt"


# Initialize the tool with a specific PDF path for exclusive search within that document
tool = TXTSearchTool(txt=directory)

with open(directory, "r", encoding="utf-8") as file:
    content = file.read()
    
# /mnt/iusers01/fse-ugpgt01/compsci01/n66425sa/hack/JD-MLPipeline/task4/knowledge/nrc_aicomplaince.pdf')

# Define an agent for text extraction
consolidator_agent = Agent(
    role="Checklist Consolidator",
    goal="Combine multiple checklists into one comprehensive checklist.",
    backstory=(
        "You are an expert in regulatory compliance and in synthesizing information from various sources. "
        "Your task is to analyze several checklists and consolidate the key points into a single, well-organized checklist."
    ),
    verbose=True,
    allow_delegation=False,
)

consolidation_task = Task(
    description=(
        "Analyze the following extracted checklists: {extracted_text}\n\n"
        "Your task is to synthesize all the key points from these checklists into one comprehensive, consolidated compliance checklist "
        "for implementing AI solutions in nuclear plant operations. Present the final checklist as a clear, well-organized bullet-point list with appropriate headings."
        "Implement in below format:"
        ),
    expected_output="A final, consolidated compliance checklist for implementing AI solutions in nuclear plants.(Do not use symbols at all in output.)",
    agent=consolidator_agent
)

# Create a Crew to run the tasks sequentially
crew = Crew(
    agents=[consolidator_agent],
    tasks=[consolidation_task],
    verbose=True,
    process=Process.sequential
)


# Run the extraction task first
extraction_results = crew.kickoff(inputs={"extracted_text": content})


with open("summary.txt", "w", encoding="utf-8") as file:  # Use "a" for append mode
    file.write(str(extraction_results))  # Add 3 newlines before writing


