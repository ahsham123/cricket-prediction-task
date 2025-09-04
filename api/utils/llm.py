from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(system_prompt: str, user_prompt: str,model='gpt-4.1',temperature=0.7,response_format=None):
    """
    Generate a response from the LLM
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if response_format:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )
        return response.choices[0].message.parsed.dict()
    
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    return response.choices[0].message.content