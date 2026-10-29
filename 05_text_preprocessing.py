
# %% Part 1: Import libraries
import re
import csv
from transformers import AutoTokenizer
from experiment import load_config, seed_everything, project_path

cfg = load_config()
seed_everything(cfg["seed"], deterministic=cfg["runtime"]["deterministic"])
DATA_DIR = project_path(cfg, "data_dir")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = cfg["models"]["text_name"]


# %% Part 2: Example radiology reports
# Synthetic examples for learning only
reports = [
    {
        "report_id": "demo_001",
        "report": """
        FINDINGS: The lungs are clear.
        No pleural effusion or pneumothorax.
        IMPRESSION: No acute cardiopulmonary abnormality.
        """
    },
    {
        "report_id": "demo_002",
        "report": """
        FINDINGS: Right lower lobe opacity is present.
        IMPRESSION: Findings suggest pneumonia.
        """
    },
    {
        "report_id": "demo_003",
        "report": """
        FINDINGS: Mild cardiomegaly and small left
        pleural effusion.
        IMPRESSION: Cardiomegaly with pleural effusion.
        """
    }
]

def clean_report(text):
    # Remove unnecessary spaces and line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for item in reports:
    item["report"] = clean_report(item["report"])

# Save demo reports
csv_path = DATA_DIR / "reports_demo.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["report_id", "report"]
    )
    writer.writeheader()
    writer.writerows(reports)

print("Saved:", csv_path)
print("Cleaned report:", reports[0]["report"])


# %% Part 3: Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts = [item["report"] for item in reports]

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=cfg["text"]["demo_max_length"],
    return_tensors="pt"
)

print("Input IDs shape:", encoded["input_ids"].shape)
print("Attention Mask shape:", encoded["attention_mask"].shape)

# Observe tokens of the first report
tokens = tokenizer.convert_ids_to_tokens(
    encoded["input_ids"][0].tolist()
)

print("First 20 tokens:", tokens[:20])
print("First 20 token IDs:",
      encoded["input_ids"][0][:20].tolist())
print("Valid token counts:",
      encoded["attention_mask"].sum(dim=1).tolist())

# %%
