from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain.messages import ToolMessage, SystemMessage
from models.models import m_qwen3_4b


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom message."""
    try:
        return handler(request)
    except Exception as e:
        # return a custom error to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


@tool("job_description_parser")
def read_job_description_from_url(job_post: str) -> str:
    """Using WebBaseLoader, crawls a job post from a given url

    Args:
        job_post: Path to the job post URL
    """
    loader = WebBaseLoader(job_post)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


tools = [read_job_description_from_url]

system_prompt = SystemMessage(
    content=[
        {
            "type": "text",
            "text": """You are a intelligent and smart recruiter.
                    You expertly analyze job descriptions. You consult with hiring manager and team to understand job post and its expectations.
                    Specifically, you find out important keywords, project experience, hands on experience as well as
                    motivation required for the role.
    """,
        },
        {
            "type": "text",
            "text": """
            Your task is to read a job post and understand all the functional and non-functional requirements required in the job.
            These functional and non-functional requirements will be used to analyze a candidate's resume for compatibility.

            #Goal
            List down all the functional and non-functional requirement in bullet points.
            """,
        },
    ]
)


def job_desc_parser_agent(query: str):
    agent = create_agent(
        model=m_qwen3_4b,
        tools=tools,
        middleware=[handle_tool_errors],
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"]["-1"].content
