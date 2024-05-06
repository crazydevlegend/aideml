"""Backend for Corcel API."""

import json
import logging
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list
from funcy import notnone, select_values


import requests

# ... other imports and setup ...

logger = logging.getLogger("aide")


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[str, float, str, dict]:
    # Setup Corcel API client (if necessary)
    api_key = "<your-api-key>"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Filter and prepare the kwargs for Corcel API
    filtered_kwargs: dict = select_values(notnone, model_kwargs)

    # Construct the message payload
    messages = opt_messages_to_list(system_message, user_message)
    payload = {
        "messages": messages,
        # Add other necessary fields according to Corcel's API requirements
        "model": "cortext-ultra",
        "stream": False,
        "top_p": 1,
        "temperature": 0.0001,
        "max_tokens": 4096,
        **filtered_kwargs,
    }

    # Measure the request time
    t0 = time.time()

    # Make the API call to Corcel
    response = requests.post(
        "https://api.corcel.io/v1/text/cortext/chat", headers=headers, json=payload
    )

    req_time = time.time() - t0

    # Initialize output variables
    output_content = None
    finish_reason = None
    info = {}

    # Check if the request was successful
    if response.status_code == 200:
        # Process the Corcel API response
        data = response.json()
        if data and data[0].get("choices"):
            first_choice = data[0]["choices"][0]
            if "delta" in first_choice:
                output_content = first_choice["delta"].get("content")
                finish_reason = first_choice.get("finish_reason")
                info = {"finish_reason": finish_reason, **payload}
    else:
        # Handle errors or unsuccessful responses
        print(f"Corcel API request failed with status code {response.status_code}")

    # Return the processed output along with additional information
    return output_content, req_time, None, None, info
