# QGpT: Improving Table Retrieval with Question Generation from Partial Tables

This repository contains the source code, corpora, and model prompts for the paper:  
📄 [**Improving Table Retrieval with Question Generation from Partial Tables**](https://openreview.net/forum?id=Q8HOV0UMwA) (TRL Workshop @ ACL 2025)


> We propose **QGpT**, a simple yet effective framework that improves open-domain table retrieval by generating synthetic questions from partial tables.

---

## 🆕 Updates (25/08/08)

We have released a original Question to gold-table datasets that we used to test recall@k:

➡️ **Hugging Face Dataset:** [cc3374twa/QGPT](https://huggingface.co/datasets/cc3374twa/QGPT)

This dataset contains question-table pairs from the original datasets, and is structured into multiple subsets (e.g., `E2E-WTQ`).

To load the dataset using `datasets`:

```python
from datasets import load_dataset

# Load the E2E-WTQ subset
dataset = load_dataset("cc3374twa/QGPT", name="E2E-WTQ", split="test")
```

## 🗂️ Repository Contents

- 📁 `Corpora/`  
  Table corpora used in each experimental table in the paper. Each subfolder corresponds to one experiment section:
  - `Table1_mimo_table_length_variation/`
  - `Table3_mimo_en_table_representation/`
  - `Table5_Single_Table_Retrieval/`
  - `Table6_Multi_Table_Retrieval/`
  - `Table7_OTTQA/`

- 📁 `prompt/`  
  Prompt templates for question generation and query decomposition (MMQA).
  - `llama3-8b-Instruct_header_extract_and_QG.txt`
  - `llama3-8b-Instruct_QG_only.txt`
  - `MMQA_query_decomposition.txt`

---

## 📦 Dataset Construction

The table corpora under `Corpora/` are preprocessed and embedded based on our proposed method.  
They are indexed using either:

- **Milvus** → [https://milvus.io](https://milvus.io)  
- **RAGatouille** → [https://github.com/AnswerDotAI/RAGatouille](https://github.com/AnswerDotAI/RAGatouille)

---

## 🧪 Reproducibility

- All table corpora are constructed based on the same datasets used in the paper.
- Each folder maps to the exact experimental tables (e.g., Table 1, Table 5).

⚠️ If any released data differs from what’s reported in the paper due to human error, please contact us at [cc3374twa@gmail.com](mailto:cc3374twa@gmail.com).

---

## 🚧 TODO

- [ ] Add setup instructions for:
  - Milvus
  - RAGatouille
- [ ] Provide a `requirements.txt` file for dependencies.
- [ ] Include example retrieval script with sample query.

---
## 📄 Citation

If you find this repository or its data useful, citing our paper would be appreciated:

```bibtex
@inproceedings{
liang2025improving,
title={Improving Table Retrieval with Question Generation from Partial Tables},
author={Hsing-Ping Liang and Che-Wei Chang and Yao-Chung Fan},
booktitle={The 4th Table Representation Learning Workshop at ACL 2025},
year={2025},
url={https://openreview.net/forum?id=Q8HOV0UMwA}
}