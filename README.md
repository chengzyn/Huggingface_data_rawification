# Huggingface Data Rawification

This project processes datasets from the Hugging Face Hub, converting Parquet files into JSONL format and adding additional keys for downstream tasks.

## Features

- Download specific subsets of datasets from the Hugging Face Hub.
- Convert Parquet files to JSONL format.
- Add a `text` key to JSONL files for easier processing.
- Recursively process directories while preserving structure.

---

## Setup Instructions

### 0. Prerequisites

Before setting up the environment, ensure you have the following installed:

- uv (for setting up the environment)
- git (for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/chengzyn/Huggingface_data_rawification.git
cd Huggingface_data_rawification
```

### 2. Use uv to set up environment

```bash
uv sync
```

## Running the Code

In `Huggingface_data_rawification/`, run the example code (after setting the configs you need in the main function)

```bash
uv run src/rawification_example_.py
```
