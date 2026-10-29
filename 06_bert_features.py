
# %% Part 1: Load reports and BERT
import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from experiment import load_config, seed_everything, get_device, project_path

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])
DATA_DIR = project_path(cfg, "data_dir")

MODEL_NAME = cfg["models"]["text_name"]

device = get_device(cfg)

with open(DATA_DIR / "reports_demo.csv",
          encoding="utf-8") as f:
    reports = list(csv.DictReader(f))

report_ids = [r["report_id"] for r in reports]
texts = [r["report"] for r in reports]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model = model.to(device)
model.eval()

print("Device:", device)
print("Number of reports:", len(texts))


# %% Part 2: Tokenize and encode
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=cfg["text"]["demo_max_length"],
    return_tensors="pt"
).to(device)

with torch.inference_mode():
    outputs = model(**inputs)

# Token-level representations
hidden_states = outputs.last_hidden_state

print("Hidden states:", hidden_states.shape)


# %% Part 3: Mean pooling
# Exclude padding tokens when averaging
mask = inputs["attention_mask"].unsqueeze(-1)
mask = mask.to(hidden_states.dtype)

text_features = (
    (hidden_states * mask).sum(dim=1)
    / mask.sum(dim=1).clamp(min=1)
)

print("Text features:", text_features.shape)
print("First report features:",
      text_features[0, :10])


# %% Part 4: Save features
features_np = text_features.cpu().numpy()

np.savez_compressed(
    DATA_DIR / "bert_demo_features.npz",
    report_ids=np.array(report_ids),
    features=features_np
)

print("Saved text features.")
