from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from src.reportgeneratorapp.llms.GroqLLM import get_llm
from langchain_core.messages import HumanMessage,SystemMessage 
from tavily import TavilyClient

def WikiSearchContent(query):
    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
    result = wiki.run(query)
    llm = get_llm()
    prompt = [
    SystemMessage(
        content=f"""You are good content writer and also a researcher.
        Follow the below instructions while generatin response for the topic: {query}
         **Instructions:**
        - Each method should include:
            - A Title heading (Bold)
            • A detailed summary (~500 words)
            • Relevant equations
            • A separator (e.g., "---") at the end
        \n\n
        if found relevant use the below extra content
        \n\n
        {result}
        
        Note: Dont include any Subheadings!!!! just the content as paragraph is needed."""
    )
]
    res = llm.invoke(prompt,reasoning_format="hidden")
    return res.content


def TavilySearchContent(query,top_k):
    client = TavilyClient()
    results = client.search(
    query = query,
    include_domains=[
        "google.com"
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "ieee.org",
        "mdpi.com",
        "researchgate.net",
        "pubmed.ncbi.nlm.nih.gov",
        "jamanetwork.com",
        "frontiersin.org",
        "hindawi.com",
    ],
    search_depth="advanced",       # Enables more comprehensive and scholarly search
    max_results=top_k,                # Limit to 15 high-quality results
    time_range="year",             # Focus on publications from the past year
    include_answer=True,           # Return a concise summary/answer if available
    include_images=False,          # Skip irrelevant images
    include_raw_content=True    # Include raw text for further processing or embedding
)

# Print the results
    extra_info = []
    for result in results['results']:
        extra_info.append(result['content']+" with a score of "+str(result['score']))
    extra_info = "\n\n".join(extra_info)
    return extra_info