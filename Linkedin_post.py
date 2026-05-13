from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["topic", "tone", "target"],
    template="""
You are an AI LinkedIn Post Generator.

Generate a LinkedIn post based on:

Topic: {topic}
Tone: {tone}
Target Audience: {target}

Requirements:
- Keep the post under 150 words
- Add relevant emojis
- Start with a strong hook
- Make it professional and engaging
- End with a CTA or question
- Suitable for LinkedIn audience
"""
)

# User Input
topic = input("Enter Topic: ")
tone = input("Enter Tone: ")
target = input("Enter Target Audience: ")

# Create Final Prompt
final_prompt = prompt.format(
    topic=topic,
    tone=tone,
    target=target
)

# Generate Response
response = llm.invoke(final_prompt)

# Print Output
print("\nGenerated LinkedIn Post:\n")
print(response.content)