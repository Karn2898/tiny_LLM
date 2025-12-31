import torch
import json
import langchain
from langchain_core.language_models import LLM

from typing import Optional, List, Any
from model_architecture import GPTLanguageModel


class MyCustomLLM(LLM):

    model: Any = None
    stoi: dict = None
    itos: dict = None
    block_size: int = None
    device: str = "cpu"

    def __init__(self, weight_path, config_path):
        super().__init__()

        print("Loading Configuration...")
        with open(config_path, 'r') as f:
            data = json.load(f)

        self.stoi = data['stoi']
        self.itos = {int(k): v for k, v in data['itos'].items()}

        config = data['config']
        self.block_size = config['block_size']

        print(" Rebuilding Model Architecture...")

        # Initialize model on CPU first
        self.model = GPTLanguageModel(**config)

        print(" Loading Weights...")


        try:
            # Load state dict to CPU first
            state_dict = torch.load(weight_path, map_location='cpu')


            if any(param.is_meta for param in self.model.parameters()):
                print(" Detected meta tensors, materializing...")
                device = torch.device(self.device)
                self.model = self.model.to_empty(device=device)

            # Load the state dict
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()

            print(" Model loaded successfully!")

        except Exception as e:
            print(f" Error loading weights: {e}")
            raise

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        input_ids = torch.tensor(
            [self.stoi.get(c, 0) for c in prompt],
            dtype=torch.long
        ).unsqueeze(0)


        with torch.no_grad():

            generated_ids = self.model.generate(input_ids, max_new_tokens=100)


        generated_text = ''.join([self.itos.get(idx.item(), '') for idx in generated_ids[0]])

       
        response = generated_text[len(prompt):]

        return response

    @property
    def _llm_type(self) -> str:
        return "custom_gpt"
