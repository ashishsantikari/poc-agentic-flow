from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.tools import tool
from langchain.messages import ToolMessage, SystemMessage
from models.models import m_liquid_lfm2


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


@tool("pdf_parser")
def parse_pdf_to_text(pdf_path: str) -> str:
    """Extracts all text from a PDF file given its local path.

    Args:
        pdf_path: the path to the pdf file
    """
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    return "\n\n".join(doc.page_content for doc in docs)


tools = [parse_pdf_to_text]

system_prompt = SystemMessage(
    content=[
        {
            "type": "text",
            "text": """You are a intelligent and smart recruiter.
                    You suggest important insights about a candidate by looking at their resume.
                    Specifically, you find out important keywords, project experience, hands on experience as well as
                    interests of the candidate.
    """,
        },
        {
            "type": "text",
            "text": """You have to list down all the important hard skills and soft skills.
            You also have to list down the different domains that the candidate has worked on and their impact.
            """,
        },
    ]
)


def resume_parser_agent(query: str):
    agent = create_agent(
        model=m_liquid_lfm2,
        tools=tools,
        middleware=[handle_tool_errors],
        system_prompt=system_prompt,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
