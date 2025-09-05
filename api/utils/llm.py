from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_response(system_prompt: str, user_prompt: str, model='gpt-4.1', temperature=0.7, response_format=None):
    """Call the OpenAI Chat Completions API and return the response content.

    Parameters
    ----------
    system_prompt : str
        The system instruction to guide the assistant's behavior.
    user_prompt : str
        The user content with task specifics (e.g., inputs and prediction).
    model : str, default 'gpt-4.1'
        Model identifier to use for generation.
    temperature : float, default 0.7
        Sampling temperature; higher values increase randomness.
    response_format : Any, optional
        If provided, uses the structured `parse` endpoint and returns a
        parsed object; otherwise returns the raw message content string.

    Returns
    -------
    str | dict
        String content when `response_format` is None, else a parsed dict.
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