from langchain_openai import ChatOpenAI


m_qwen3_4b = ChatOpenAI(
    base_url="http://localhost:1234/v1/",
    model="qwen/qwen3-4b-2507",
    temperature=0.6,
    api_key="",
)


m_liquid_lfm2 = ChatOpenAI(
    base_url="http://localhost:1234/v1/",
    model="liquid/lfm2-1.2b",
    temperature=0.6,
    api_key="",
)
