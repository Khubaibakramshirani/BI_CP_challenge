import os

from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()
print(os.environ.get("OPENAI_API_KEY"))

# Define PDF paths
PRESENTATION_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/presentation.pdf"))
PROXY_STATEMENT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/proxy_statement.pdf"))

print("presentation", PRESENTATION_PATH)