from crewai import Agent, Task, Crew, Process, LLM
import os
from crewai_tools import PDFSearchTool

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

directory = "/mnt/iusers01/fse-ugpgt01/compsci01/n66425sa/hack/JD-MLPipeline/task4/knowledge/"

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    if os.path.isfile(file_path):  # Ensures it's a file
        print(f"Processing file: {file_path}")
        

        # Initialize the tool with a specific PDF path for exclusive search within that document
        tool = PDFSearchTool(pdf=file_path)

# /mnt/iusers01/fse-ugpgt01/compsci01/n66425sa/hack/JD-MLPipeline/task4/knowledge/nrc_aicomplaince.pdf')

        # Define an agent for text extraction
        extractor_agent = Agent(
            role="Text Extractor",
            goal="Extract the full regulatory guidelines text from the provided PDF.",
            backstory="You are an expert in extracting and summarizing text from complex regulatory documents.",
            verbose=True,
            allow_delegation=False,
        )

        extraction_task = Task(
            description=("Extract and summarize (if needed) the key regulatory guidelines from the PDF document. "
                        "covering guidance for implementing AI solutions in the nuclear lifecycle."),
            expected_output="A plain text summary (or full text) of the regulatory guidelines extracted from the PDF.",
            agent=extractor_agent
        )

        # Define an agent for generating a compliance checklist
        checklist_agent = Agent(
            role="Checklist Generator",
            goal=("Analyze the extracted regulatory guidelines and generate a comprehensive compliance checklist "
                "for implementing AI solutions in nuclear plant operations."),
            backstory=("You are an expert in regulatory compliance and Natural Language Processing. "
                    "Your task is to produce a clear, bullet-point checklist based on provided regulatory guidelines."),
            verbose=True,
            allow_delegation=False,
        )

        checklist_task = Task(
            description=("Use generative AI & NLP to analyze the following extracted regulatory guidelines: {extracted_text} "
                        "and develop a comprehensive checklist/compliance rule for implementing AI solutions into the nuclear lifecycle. "
                        "The checklist should be presented as a well-organized bullet-point list with clear headings."),
            expected_output="A final, consolidated compliance checklist for AI solutions in nuclear plants.",
            agent=checklist_agent
        )

        # Create a Crew to run the tasks sequentially
        crew = Crew(
            agents=[extractor_agent, checklist_agent],
            tasks=[extraction_task, checklist_task],
            verbose=True,
            process=Process.sequential
        )


        # Run the extraction task first
        extraction_results = crew.kickoff(inputs={})
        # print("extraction results: ", extraction_results)
        # print("extraction results: ", str(extraction_results))
        print("Extracted Regulatory Guidelines:")
        print(extraction_results)

        # Then, run the checklist generation task by passing the extracted text
        checklist_results = crew.kickoff(inputs={"extracted_text": str(extraction_results)})
        print("\nFinal Compliance Checklist:")
        print(checklist_results)

        with open("checklist.txt", "a", encoding="utf-8") as file:  # Use "a" for append mode
            file.write("\n\n\n" + str(checklist_results) + "\n\n")  # Add 3 newlines before writing
