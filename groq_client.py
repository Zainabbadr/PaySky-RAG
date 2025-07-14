import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    return Groq(api_key=api_key)

def ask_groq(client, context: str, question: str):
    prompt = f"""
You are an expert data analyst assistant. Use the following data records to answer the question provided.

Instructions:
- Analyze the data provided in the context
- Give specific insights based on the actual data
- If asked about trends, patterns, or statistics, provide concrete examples from the data
- Be professional and helpful
- If the question cannot be answered from the provided context, politely explain what information is available
- Adapt your analysis style to the type of data provided (business, academic, operational, etc.)
- If the user is greeting or thanking, respond appropriately

Context (Data Records):
{context}

Question:
{question}

Answer:"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a data analytics assistant. Answer based on the context provided and adapt to any type of tabular data."},
            {"role": "user", "content": f"Context:\n{context}"},
            {"role": "user", "content": f"Question:\n{question}"}
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
    )
    # Stream the output into a string
    answer = ""
    for chunk in completion:
        answer += chunk.choices[0].delta.content or ""
    return answer
