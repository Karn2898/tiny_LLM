import os
import json
import requests


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "..", "data", "instruction_data.json")


with open(file_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
def format_input(entry):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
import requests

def query_model(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(url, json=data, timeout=120)
    result = r.json()

    if "response" in result:
        return result["response"]

    if "message" in result and "content" in result["message"]:
        return result["message"]["content"]

    raise ValueError(f"Unexpected response format: {result}")

for entry in test_data[:3]:
    model_response = query_model(format_input(entry))

    prompt = (
        f"Given the input `{format_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{model_response}` "
        f"on a scale from 0 to 100, where 100 is the best score."
    )

    print("\nDataset response:")
    print(">>", entry["output"])

    print("\nModel response:")
    print(">>", model_response)

    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n-------------------------")
