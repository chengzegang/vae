from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub import whoami, create_repo
import os

creator_name = "Zegang Cheng"
github_repo = "https://github.com/chengzegang/vae"

card_data = ModelCardData(language="en", license="mit", model_name="vae")
content = f"""
---
{ card_data.to_yaml() }
---
# Model Card for VAE

This is an PyTorch implementation of Variational AutoEncoder, model structure was inspired by the one used with Stable-Diffusion.

This model created by [@{creator_name}]({github_repo}).
"""

card = ModelCard(content)
card.repo_type = "model"
card.card_data_class = ModelCardData
card.validate("model")
card.save(os.path.join(os.path.dirname(__file__), "model_card.md"))
