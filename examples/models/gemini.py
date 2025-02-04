import asyncio
import os

from dotenv import load_dotenv

from browser_use import LLM, Agent

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

llm = LLM(model="gemini/gemini-2.0-flash-exp")


async def run_search():
    agent = Agent(
        task=(
            'Go to url r/LocalLLaMA subreddit and search for "browser use" in the search bar and click on the first post and find the funniest comment'
        ),
        llm=llm,
        max_actions_per_step=4,
    )

    await agent.run(max_steps=25)


if __name__ == "__main__":
    asyncio.run(run_search())
