import psutil

def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
print("Ollama running:", check_if_running("ollama"))

import requests


def query_model(
        prompt,
        model="tinyllama",
        url="http://localhost:11434/api/chat"  # Changed port back to default 11434
):

    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    response = requests.post(url, json=data)

    # Check for errors
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        response.raise_for_status()

    return response.json()['message']['content']

model = "tinyllama"
result = query_model("What do Llamas eat?", model)
print(result)


