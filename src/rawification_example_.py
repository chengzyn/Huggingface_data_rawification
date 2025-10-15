import os
from huggingface_hub import snapshot_download
import json
from tqdm import tqdm
import os
import polars as pl
from tqdm import tqdm


def examine_parquet_file(parquet_file, index=None):
    """
    Looks at the schema of the parquet file and examines a particular index (if specified).

    This allows us to view the keys and data type within the parquet file.

    """
    # Load the Parquet file into a Polars DataFrame
    df = pl.read_parquet(parquet_file)

    # Print the schema of the DataFrame
    print(df.schema)

    if index is not None:
        # Check if the index is within the range of the DataFrame
        if index < 0 or index >= df.height:
            print(f"Index {index} is out of range. DataFrame has {df.height} rows.")
            return

        # Get the row at the specified index
        row = df.row(index)

        # Print the row
        print(f"Row at index {index}: {row}")


def parquet_to_jsonl(parquet_file, output_jsonl):
    """
    Converts a Parquet file to a JSONL file.

    Args:
        parquet_file (str): Path to the Parquet file.
        output_jsonl (str): JSONL file
    """
    # Load the Parquet file into a Polars DataFrame
    df = pl.read_parquet(parquet_file)

    # Write the DataFrame to a JSONL file
    df.write_ndjson(output_jsonl)


def rawify_text(parquet_dir, output_dir):
    """
    Recursively process all Parquet files in the directory and its subdirectories,
    and write them as JSONL files in the output directory, preserving the directory structure.

    Args:
        parquet_dir (str): Path to the root directory containing Parquet files.
        output_dir (str): Path to the root directory where JSONL files will be saved.
    """
    # Traverse the directory tree
    for root, _, files in os.walk(parquet_dir):
        for file in tqdm(files, desc="Processing Parquet files"):
            if file.endswith(".parquet"):
                parquet_path = os.path.join(root, file)

                # Preserve the relative directory structure in the output directory
                relative_path = os.path.relpath(root, parquet_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Define the output JSONL file path
                output_jsonl_path = os.path.join(output_subdir, file.replace(".parquet", ".jsonl"))

                try:
                    parquet_to_jsonl(parquet_path, output_jsonl_path)
                except Exception as e:
                    print(f"Error processing {parquet_path}: {e}")


def download_data(download_dir, subset=None):
    """
    Downloads a specific subset of the dataset from the Hugging Face Hub.

    Args:
        download_dir (str): The directory where the dataset will be downloaded.
        subset (str, optional): The specific subset of the dataset to download.
                                If provided, only files matching the subset pattern will be downloaded.

    Functionality:
        - Uses the `snapshot_download` function from the Hugging Face Hub to download the dataset.
        - Downloads the dataset to the specified `download_dir`.
        - If a `subset` is provided, only files matching the pattern `<subset>/*` will be downloaded.
        - Ensures that the downloaded files are stored locally in the specified directory.
    """
    # snapshot download all the data
    snapshot_download(
        "nvidia/Nemotron-CC-Math-v1",
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=[f"{subset}/part_000000.parquet"],
    )


def add_text_key_to_jsonl(input_jsonl, output_jsonl):
    """
    Adds a 'text' key to each JSON object in a JSONL file.

    Args:
        input_jsonl (str): Path to the input JSONL file.
        output_jsonl (str): Path to the output JSONL file.
    """
    with open(input_jsonl, "r", encoding="utf-8") as infile, open(output_jsonl, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                # Parse the JSON object
                json_obj = json.loads(line.strip())

                # Create the 'text' key
                json_obj["text"] = f"{json_obj['input']}\n{json_obj['output']}"

                # Write the updated JSON object to the output file
                outfile.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
            except KeyError as e:
                print(f"Missing key {e} in line: {line}")


def add_text_key_to_jsonl_directory(input_dir, output_dir):
    """
    Adds a 'text' key to each JSON object in all JSONL files in a directory.

    Args:
        input_dir (str): Path to the directory containing input JSONL files.
        output_dir (str): Path to the directory to save the output JSONL files.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Iterate over all JSONL files in the input directory
    for filename in tqdm(os.listdir(input_dir), desc="Processing JSONL files"):
        if filename.endswith(".jsonl"):  # Process only .jsonl files
            input_jsonl = os.path.join(input_dir, filename)
            output_jsonl = os.path.join(output_dir, filename)
            add_text_key_to_jsonl(input_jsonl, output_jsonl)


if __name__ == "__main__":
    # config
    do_download = True  # set to False after you complete download
    examine_file = False  # set to True if you want to examine a parquet file
    examine_index = None  # index of the row to examine in the parquet file
    do_rawify = False  # set to True to rawify the dataset

    # paths
    root_dir = "/home/czyn/projects/sandbox"
    os.makedirs(
        download_dir := os.path.join(root_dir, "nemotron_cc_math_v1"),
        exist_ok=True,
    )
    os.makedirs(
        output_root_dir := os.path.join(root_dir, "nemotron_cc_math_v1", "jsonl_output"),
        exist_ok=True,
    )

    # Download the dataset (Parquet files) - commented out after you complete download
    if do_download:
        # loop thru the selected snapshots that we want to download
        subsets = ["3", "4plus"]
        for subset in subsets:
            print(f"Downloading snapshot version: {subset}")
            download_data(download_dir, subset=subset)

    # Examine a parquet file
    if examine_file:
        sample_parquet_file = "/mnt/work/chengzyn/data/nemotron_cc_math_v1/data/CC-MAIN-2024-22/000_00000.parquet"
        examine_parquet_file(sample_parquet_file, index=examine_index)

    # Rawification (Parquet to JSONL)
    parquet_dir = os.path.join(download_dir, "data")
    output_dir = os.path.join(output_root_dir)
    os.makedirs(output_dir, exist_ok=True)
    if do_rawify:
        rawify_text(parquet_dir, output_dir)
