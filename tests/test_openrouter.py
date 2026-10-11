
from dotenv import load_dotenv
#from crewai import LLM

from src.multi_agent_ai_deep_researcher.llm_config import get_llm

load_dotenv()


def main() -> None:
    llm = get_llm()

    response = llm.call(
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with exactly: "
                    "OpenRouter connection successful"
                ),
            }
        ]
    )

    print("Model response:", response)


if __name__ == "__main__":
    main()
