
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if not api_key or api_key == "your_openrouter_api_key":
        print("ERROR: Set your actual OpenRouter key in .env.")
        return

    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print("Authentication successful.")
            print("HTTP status:", response.status)
            print("Key details:", response.read().decode())

    except urllib.error.HTTPError as error:
        print("Authentication check failed.")
        print("HTTP status:", error.code)
        print(error.read().decode())

    except urllib.error.URLError as error:
        print("Network error:", error.reason)


if __name__ == "__main__":
    main()
