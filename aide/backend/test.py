from backend_corcel import guarded_cortex_api, guard
from guardrails import Guard
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    date_of_birth: int
    job: str

prompt = """
   ${prompt}

    ${gr.complete_json_suffix_v2}
"""

guard = Guard.from_pydantic(output_class=Person, prompt=prompt)
raw,validated_output,*rest = guard(
    guarded_cortex_api,
    prompt=prompt,
    instruction="describe obama",
    msg_history=[{"role": "system", "content": "answer question"}]
)
print(validated_output)

