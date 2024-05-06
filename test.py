import logging
from typing import cast
from aide.backend import backend_anthropic, backend_openai, backend_corcel
from aide.backend.utils import (
    FunctionSpec,
    OutputType,
    PromptType,
    compile_prompt_to_md,
)


logger = logging.getLogger("aide")


def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    print(f"Querying model {model} with kwargs {model_kwargs}")

    query_func = (
        backend_corcel.query
        if "cortext" in model
        else (backend_openai.query if "gpt-" in model else backend_anthropic.query)
    )

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message="You should determine if there were any bugs as well as report the empirical findings.",
        user_message=compile_prompt_to_md(user_message) if user_message else None,
        func_spec=func_spec,
        **model_kwargs,
    )

    return output


if __name__ == "__main__":
    prompt = {
        "Introduction": (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        ),
        "Task description": "example_tasks/house_prices",
        "Implementation": None,
        "Execution output": None,
    }

    # Run the query function
    responses = cast(
        dict,
        query(
            system_message=prompt,
            user_message=None,
            func_spec=None,
            model="cortext-ultra",
            temperature=0.0001,
            max_tokens=4096,
        ),
    )

    print(responses)
