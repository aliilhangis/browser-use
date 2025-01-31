import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import SecretStr

from browser_use import Agent

load_dotenv()
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError('GROQ_API_KEY is not set')

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=SecretStr(api_key),
    temperature=0.0
)

async def run_search():
    agent = Agent(
        task=(
            'Go to url r/LocalLLaMA subreddit and search for "browser use" '
            'in the search bar and click on the first post'
        ),
        llm=llm,
        use_vision=False
    )

    await agent.run(max_steps=25)

if __name__ == '__main__':
    asyncio.run(run_search()) 