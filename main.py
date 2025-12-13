from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from models.models import m_qwen3_4b
from agents.job_desc_parser.job_desc_parser import job_desc_parser_agent
from agents.resume_parser.resume_parser import resume_parser_agent

@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom message."""
    try:
        return handler(request)
    except Exception as e:
        # return a custom error to the model
        print(f"\033[2;31;43m {e}")
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


@tool(
    "job_desc_parser_agent",
    description="Parses a job description. If url is provided, fetches the job description",
)
def call_job_description_parser_agent(query: str):
    return job_desc_parser_agent(query)


@tool("resume_parser_agent", description="Parses and Summarizes a resume")
def call_resume_parser_agent(query: str):
    return resume_parser_agent(query)


system_prompt = SystemMessage(
    content=[
        {
            "type": "text",
            "text": """You are a intelligent and smart recruiter.
            You suggest the candidate  a real comparision of their resume with the given job description.
            The user may provide the resume and/or the job description in text format or provide you the pdf for the resume,
            or the direct link to job description.
    """,
        },
        {
            "type": "text",
            "text": """You have to strictly compare the job requirement and the resume. You should provide the following
            information for transparency
            - summary - give an honest summary of the comparision. use a scale of 1 to 10. 1 being the no match and 10 being perfect match.
            - positive - positive highlights that are common across the resume and the jon description.
            - negative or missing - skills which are missing and will impact your working if chosen for the job
            - compatibility for the role - does it match the expectation vs the actual. for example an individual contributor will not fit in
            for a team lead role where there is more managerial work and so on.
            """,
        },
    ]
)


def initialize_agent():
    agent = create_agent(
        model=m_qwen3_4b,
        middleware=[handle_tool_errors],
        tools=[call_resume_parser_agent, call_job_description_parser_agent],
        system_prompt=system_prompt,
    )
    return agent


def main():
    agent = initialize_agent()
    result = agent.invoke(
        {
            "messages": [
                AIMessage("""
                  Hello, my name is Scanner The recruiter and I will help you with finding out your potential job compatibility.
                Think of me as your relationship advisor but for job. Provide me with your resume and the job description.
                Resume and Job Description can be texts, but if you are feeling lazy, provide me the link to job description.
                  """),
                HumanMessage(
                    """Hi my name is Ashish. My resume is present in the path "docs/resume.pdf".
                The job is posted on https://www.stepstone.de/stellenangebote--Senior-Software-Engineer-m-f-d-Berlin-Merantix-AG--13103306-inline.html
                """
                ),
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
