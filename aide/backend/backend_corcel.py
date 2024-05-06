from guardrails import Guard
import requests
from funcy import notnone, select_values
from utils import opt_messages_to_list
from typing import Optional
import json
# Create a Guard class
guard = Guard()

def cortex_api(
    prompt: Optional[str] = None,
    instruction: Optional[str] = None,
    msg_history: Optional[list[dict]] = None,
    **kwargs
) -> str:
    """Custom LLM API wrapper.

    At least one of prompt, instruction or msg_history should be provided.

    Args:
        prompt (str): The prompt to be passed to the LLM API
        instruction (str): The instruction to be passed to the LLM API
        msg_history (list[dict]): The message history to be passed to the LLM API
        **kwargs: Any additional arguments to be passed to the LLM API

    Returns:
        str: The output of the LLM API
    """
  
    # Setup Corcel API client
    api_key = "b7a139c3-8a8e-4d19-a754-27724054dd2f"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Filter and prepare the kwargs for Corcel API
    filtered_kwargs: dict = select_values(notnone, kwargs)

    # Construct the message payload
    messages = opt_messages_to_list(prompt, instruction)
    if not messages and not msg_history:
        messages = [{"role": "user", "content": ""}]  # Add a default empty message

    payload = {
        "messages": messages or msg_history,
        # Add other necessary fields according to Corcel's API requirements
        "model": "cortext-ultra",
        "stream": False,
        "top_p": 1,
        "temperature": 0.0001,
        "max_tokens": 4096,
        **filtered_kwargs,
    }

    # Make the API call to Corcel
    response = requests.post(
        "https://api.corcel.io/v1/text/cortext/chat", headers=headers, json=payload
    )

    if response.status_code == 200:
        # Process the Corcel API response
        data = response.json()

        if data and data[0].get("choices"):
            first_choice = data[0]["choices"][0]
            if "delta" in first_choice:
                output_content = first_choice["delta"].get("content")
            
                # Check if the output content is a valid JSON string
                try:
                    json.loads(output_content)
                    return output_content
                except json.JSONDecodeError:
                    # If the output content is not a valid JSON string, construct a default response
                    return output_content

    # Handle errors or unsuccessful responses
    print(f"Corcel API request failed with status code {response.status_code}")
    return ''

def guarded_cortex_api(*args, **kwargs):
    return cortex_api(*args, **kwargs)