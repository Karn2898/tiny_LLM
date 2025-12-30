import chainlit as cl
import torch
import tiktoken
from pathlib import Path
from llms_from_scratch.ch04 import GPTModel
from llms_from_scratch.ch05 import generate, text_to_token_ids, token_ids_to_text
import sys
from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
)

device = torch.device( "cpu")


def get_model_and_tokenizer():

    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    model_path = Path(r"C:\Users\DELL\PycharmProjects\PythonProject4\model\gpt2-medium355M-sft.pth")
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. "
        )
        sys.exit()

    checkpoint = torch.load(model_path,
                            map_location=torch.device("cpu"),
                            weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@cl.on_message
async def main(message: cl.Message):

    torch.manual_seed(123)

    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """
    tokenizer, model, model_config = get_model_and_tokenizer()
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    await cl.Message(
        content=f"{response}",
    ).send()