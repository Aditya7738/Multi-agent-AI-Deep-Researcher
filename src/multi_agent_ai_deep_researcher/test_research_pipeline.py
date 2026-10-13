
from src.multi_agent_ai_deep_researcher.query_planner import (
    plan_research,
)
from src.multi_agent_ai_deep_researcher.contextual_retriever import (
    retrieve_sources,
)

def main() -> None:
    question = input("Enter your research question: ").strip()

    try:
        # Stage 1: Generate focused queries using OpenRouter.
        queries = plan_research(question)

        print("\nPLANNED QUERIES")
        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")

        # Stage 2: Search a bounded number of queries using Tavily.
        sources = retrieve_sources(queries)

        print("\nRETRIEVAL SUMMARY")
        print(f"Queries planned: {len(queries)}")
        print(f"Unique sources retrieved: {len(sources)}")

        if not sources:
            print("No sources retrieved.")
            return

        print("\nEVIDENCE FOR THE NEXT AGENT")
        for source in sources:
            print(f"\n[{source['source_id']}] {source['title']}")
            print(f"URL: {source['url']}")
            print(f"Content: {source['content']}")

    except (ValueError, RuntimeError) as error:
        print(f"\nResearch pipeline stopped: {error}")

if __name__ == "__main__":
    main()