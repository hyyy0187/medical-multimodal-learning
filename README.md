# Medical Image–Text Representation Learning

**A CLIP-style dual encoder for chest X-ray and radiology report retrieval.**

This project studies contrastive representation learning between chest radiographs and radiology reports. It includes cached-feature training, partial image-encoder fine-tuning, retrieval baselines, data auditing, robustness experiments, and failure analysis.

The published metrics describe an educational research prototype evaluated on a validation set. They do not establish clinical diagnostic or prognostic performance.

## Architecture

```text
Chest X-ray → ImageNet ResNet-18 → 512 features → Linear(512, 256) → L2 normalization
                                                                                ↕ cosine similarity
Report → BioClinicalBERT → masked mean pooling → 768 features → Linear(768, 256) → L2 normalization
```

- Both projection heads are trained using symmetric image-to-text and text-to-image cross-entropy, with the paired item on the similarity matrix diagonal. This is a CLIP-style contrastive objective with temperature 0.07.
- The frozen-encoder stage trains only the two projection heads: **328,192 trainable parameters**.
- Partial fine-tuning updates ResNet-18 `layer4` and both projection heads. Text features remain cached, so BioClinicalBERT is not fine-tuned.
- `MedicalCLIP` in `models.py` is a project-defined class. The image and text encoders start from separate pretrained models; the code does not load a complete pretrained medical CLIP checkpoint.
- The optional pneumonia classifier in script 04 is a separate exercise. Its saved weights are not used to initialize the retrieval experiments.

## Repository layout

```text
medical-multimodal-learning/
├── README.md
├── requirements.txt
├── requirements-lock.txt           # Historical Windows/CUDA environment snapshot
├── config.yaml
├── .gitignore
├── experiment.py                   # Configuration, seeds, paths, DataLoader helpers
├── CONFIG_GUIDE.md                 # Configuration guide in Chinese
├── models.py
├── losses.py
├── 01_pytorch_basics.py            # Optional foundational exercises, 01–04
├── 02_resnet_features.py
├── 03_dataloader.py
├── 04_train_pneumonia.py
├── 05_text_preprocessing.py        # Small synthetic-report demonstration
├── 06_bert_features.py             # Features for the synthetic reports
├── 07_text_dataloader.py           # Real IU report matching and text features
├── 08_image_text_pairs.py          # Image matching, splits, and image features
├── 09_dual_encoder_test.py         # Forward/gradient/update checks
├── 10_train_contrastive.py
├── 11_retrieval_baselines.py
├── 12_partial_finetune.py
├── 13_day7_data_audit.py
├── 14_day7_robustness.py
├── 15_day8_survival_demo.py         # Compatibility entry point
├── 16_export_splits.py
├── 17_export_project_assets.py
├── tutorials/
│   └── 15_day8_survival_demo.py     # Independent synthetic survival tutorial
├── tests/
├── results/                        # Only curated aggregate CSVs are tracked
│   ├── retrieval_comparison.csv
│   ├── day6_final_comparison.csv
│   ├── day7_robustness.csv
│   ├── day7_ablation.csv
│   ├── day7_data_audit.csv
│   ├── day7_seen_unseen_comparison.csv
│   └── training_history.csv
├── figures/
│   ├── training_curve.png
│   ├── retrieval_comparison.png
│   ├── day7_image_robustness.png
│   ├── day7_text_noise_robustness.png
│   └── day7_text_truncation.png
├── reports/
│   └── technical_report.md
├── splits/
│   ├── train_uids.csv
│   ├── val_uids.csv
│   └── test_uids.csv
└── data/                           # Local data and feature caches; not uploaded
```

Model weights, raw images and reports, feature arrays, and detailed failure-case files are excluded from version control. `results/` can contain additional local outputs, but only the seven aggregate tables listed above are intended for publication. The training history is retained because it supports the report's checkpoint-selection and overfitting analysis. `figures/` is the location for curated figures; an older local `tables/` export is not the source of truth.

## Dataset and local preparation

The main experiments use frontal images and associated reports from the Indiana University Chest X-ray / Open-i dataset. Obtain the data from its original or appropriately licensed distribution; see [Open-i data access](https://openi-vip.nlm.nih.gov/faq). The repository does not redistribute it or automatically download it.

The original Open-i release uses XML reports and images. These scripts expect the converted `indiana_reports.csv` and `indiana_projections.csv` metadata layout shown below. Obtain a compatible conversion or prepare equivalent CSVs; downloading the original release alone does not satisfy this input format. This repository does not currently include an XML-to-CSV converter.

Prepare the following local layout:

```text
data/iu_xray/
├── indiana_reports.csv             # uid, findings, impression, ...
├── indiana_projections.csv         # uid, filename, projection
└── images/
    └── images_normalized/
        └── *.png
```

Script 07 joins `findings` and `impression`, removes empty reports, retains frontal image metadata, and matches records by UID. Script 08 resolves image files and constructs UID-disjoint partitions.

The recorded local dataset contains:

| Partition | Matched image–report records | Unique report UIDs |
|---|---:|---:|
| Train | 3,039 | 2,932 |
| Validation | 383 | 367 |
| Test | 372 | 367 |
| Total | 3,794 | 3,666 |

The contrastive and retrieval protocol selects one image per UID by sorting filenames and retaining the first. Consequently, training uses 2,932 pairs and validation retrieval uses 367 pairs. UID separation is at the report/study level; it does not prove patient-level independence.

The committed `splits/*_uids.csv` files record the existing partitions. Current script 08 generates partitions from the configured seed rather than automatically loading these exported UID lists. Compare regenerated UID sets with the exports when attempting to reproduce the recorded partition.

## Installation and configuration

Use a Python 3.11 environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

For GPU execution, select a PyTorch/Torchvision build compatible with your system using the [official installation instructions](https://pytorch.org/get-started/locally/). `requirements-lock.txt` records a historical Windows environment, including CUDA-specific packages; it is not a platform-independent installation recipe. First use of the pretrained encoders may download weights from their providers.

The inspected local environment uses Python 3.11.17, PyTorch 2.11.0+cu128, Torchvision 0.26.0+cu128, Transformers 5.19.0, NumPy 2.4.6, and pandas 2.3.3. Recording these versions documents the environment; it does not mean that all historical experiments have been rerun in a clean installation.

Major parameters are centralized in [config.yaml](config.yaml):

```yaml
seed: 42

runtime:
  device: auto
  deterministic: true
  num_workers: 0

finetune:
  pilot: false
  batch_size: 8
  epochs: 3
  encoder_learning_rate: 0.00001
  projection_learning_rate: 0.0001
```

This is an excerpt; retain the other sections in the supplied configuration. Paths resolve relative to the project root. On Windows, keep `num_workers: 0` because the learning scripts contain top-level execution code. Set `finetune.pilot: true` for the smaller fine-tuning experiment and `false` for the full experiment. Reload from the first cell after changing configuration.

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for the seed, device, checkpoint, and cache behavior. This guide is intentionally retained in Chinese; the other project documentation and source comments are in English.

## Running the experiments

Run commands from the repository root, after preparing the local dataset. The scripts also retain VS Code `# %%` cell markers.

1. Extract real report features, construct pairs and partitions, and extract image features:

   ```bash
   python 07_text_dataloader.py
   python 08_image_text_pairs.py
   ```

   These create local CSV metadata and feature arrays under `data/iu_xray/`. Scripts 05 and 06 use three synthetic reports and are optional demonstrations, not prerequisites for processing IU data.

2. Optionally check the dual encoder and gradients, then train and evaluate the frozen-encoder projection model:

   ```bash
   python 09_dual_encoder_test.py
   python 10_train_contrastive.py
   python 11_retrieval_baselines.py
   ```

   Script 09 includes optimizer updates for verification; it is not a dataset-independent unit-test suite. Script 10 selects its checkpoint by validation contrastive loss. Script 11 evaluates on the validation partition.

3. Fine-tune the image encoder, audit data, and run robustness experiments:

   ```bash
   python 12_partial_finetune.py
   python 13_day7_data_audit.py
   python 14_day7_robustness.py
   ```

   Script 12 loads `results/best_projection_model.pth`, then selects the fine-tuned checkpoint by mean bidirectional validation R@5. Script 13 both audits data and writes a serialization-compatible `best_partial_finetune_fixed.pth`; its conversion step expects the locally generated `best_partial_finetune.pth`. Script 14 loads that fixed checkpoint. The legacy conversion uses unrestricted checkpoint loading and should only be used on your own trusted checkpoint.

4. Export the split identifiers and curated publication assets:

   ```bash
   python 16_export_splits.py
   python 17_export_project_assets.py
   ```

   Asset export organizes existing results; it is not a new training or evaluation run.

5. Run the independent survival tutorial or the lightweight configuration tests:

   ```bash
   python tutorials/15_day8_survival_demo.py
   python -m unittest discover -s tests -v
   ```

   The survival tutorial generates synthetic patients and artificial image/text signals. It does not load retrieval embeddings or real longitudinal outcomes, and its outputs are not clinical survival-model results.

The optional scripts 02–04 require local `chest_xray.png` and/or the separate `chest_xray/` classification dataset. Those assets are not required for the main IU retrieval pipeline.

## Recorded validation retrieval results

The following values come from the existing aggregate CSV files, not a new run performed during repository preparation. Every query is ranked against the same pool of **367 validation candidates**.

| Method | I2T R@5 (%) | T2I R@5 (%) | Mean R@5 (%) |
|---|---:|---:|---:|
| Random retrieval | 1.294 | 1.281 | 1.288 |
| Pretrained encoders + untrained projection heads | 1.907 | 0.817 | 1.362 |
| Frozen encoders + trained projection heads | 4.905 | 4.360 | 4.632 |
| Partial fine-tuning of ResNet `layer4` | **7.084** | **6.267** | **6.676** |

I2T means image-to-text retrieval; T2I means text-to-image retrieval. R@K is the percentage of queries whose specified paired item appears in the top K. Median rank (MedR), available in the CSV files, is better when lower. A different UID with identical or clinically similar report text still counts as incorrect under the current strict one-to-one protocol.

Fine-tuning increased mean validation R@5 by **2.044 percentage points**, or approximately **44.1% relative** to the frozen model. Absolute instance-level retrieval performance remains limited; fine-tuned top-1 recall is 1.36% for I2T and 1.09% for T2I. These are retrieval measures, not disease classification accuracy.

Sources: [retrieval baseline comparison](results/retrieval_comparison.csv), [final frozen/fine-tuned comparison](results/day6_final_comparison.csv), and [ablation results](results/day7_ablation.csv).

![Frozen-encoder training curve](figures/training_curve.png)

![Retrieval baseline comparison](figures/retrieval_comparison.png)

## Robustness, auditing, and limitations

- Gaussian image noise with standard deviation 0.15, added after ImageNet normalization, reduced mean R@5 from 6.676% to 3.542%.
- Truncation to 32 tokenizer tokens yielded mean R@5 of 5.586%. A 15% word-dropout setting yielded 6.812% in one fixed random realization; this does not establish a reliable improvement. Negation words are protected in the dropout experiment.
- Restoring `layer4` to its ImageNet initialization while retaining the learned projection heads yielded mean R@5 of 6.131%. This intervention also changes encoder/head compatibility, so it does not isolate a causal contribution of `layer4`.
- The performed audit found no cross-partition overlap of UIDs, filenames, or exact image-file hashes. Patient-level independence was **unverifiable**. Hash checks cover byte-identical files, not all possible near duplicates.
- There were 57 normalized report texts shared between train/validation, 54 between train/test, and 26 between validation/test. These counts refer to distinct shared texts, not numbers of patients.
- Of 367 validation report UIDs, 91 (24.8%) had normalized text seen in training. There were also 51 validation records in internally duplicated report-text groups. Seen/unseen query comparisons retain all 367 candidates and are confounded by report duplication; lower seen-report recall does not demonstrate absence of memorization.
- Training selection, robustness experiments, and the reported comparisons use the validation set. An independent test partition exists, but **no independent test-set retrieval result is claimed**.
- Perturbation results use a single fixed random realization and have no multi-seed confidence intervals. Cached clean text features and freshly encoded clean text features have slightly different rankings, so text perturbations should be compared against their own re-encoded baseline.

![Image noise robustness](figures/day7_image_robustness.png)

See [the technical report](reports/technical_report.md) and the curated CSV tables for details.

## Reproducibility scope

The repository provides experiment code, configuration, recorded split UIDs, aggregate metrics, and curated figures. Raw data, encoder downloads, trained checkpoints, and feature caches must be obtained or generated locally.

The recorded outputs were produced during earlier interactive experiments. Configuration and seed handling were subsequently centralized. Independent DataLoader generators and deterministic algorithm settings can change the random-number sequence compared with those earlier runs; the current defaults are not a guarantee of bit-for-bit reproduction of the historical tables.

Changing `seed` does not rewrite existing split CSVs. Re-run script 08 when intentionally changing the partition. Changing the text model or token limit requires regenerating text features with script 07 before the downstream experiments; the current text projection expects 768-dimensional features. When loading an existing checkpoint, scripts 11 and 12 preserve its projection dimension, and script 12 also preserves its temperature. Effective values are recorded in configuration snapshots.

The current configuration retains script 10's `val_drop_last: true`: contrastive validation loss uses 352 of 367 pairs, while full retrieval evaluation uses all 367. The two training stages also have different batch sizes and checkpoint-selection criteria, so their loss values should not be compared directly.

## References and acknowledgments

- [Indiana University Chest X-ray / Open-i data access](https://openi-vip.nlm.nih.gov/faq).
- [BioClinicalBERT model card](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT): `emilyalsentzer/Bio_ClinicalBERT`.
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- [PyTorch installation](https://pytorch.org/get-started/locally/); Torchvision, Hugging Face Transformers, scikit-learn, and lifelines.

Consult the original dataset and model documentation for citations, access conditions, and usage terms. Dataset and pretrained-model terms apply separately from the code in this repository.
