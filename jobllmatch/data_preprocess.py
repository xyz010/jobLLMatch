import re

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BartTokenizer


def data_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df.loc[:, "score"] = df["score"].astype(str).copy()

    return df


def create_dataset_splits(
    df: pd.DataFrame, train_split: float, val_split: float, random_state: int
) -> list[pd.DataFrame]:
    df_train, df_temp = train_test_split(df, test_size=train_split, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=val_split, random_state=42)
    return df_train, df_val, df_test


def create_dataset_prompt_col(df: pd.DataFrame) -> pd.DataFrame:
    df["prompt"] = (
        "<<RESUME>> "
        + df["resume"]
        + ", <<JOB>> "
        + df["job"]
        + ", <<SCORE>> "
        + df["score"]
        + ", <<REPORT>> "
        + df["report"]
    )
    df["res_and_job"] = "<<RESUME>> " + df["resume"] + ", <<JOB>> " + df["job"]
    df["score_and_report"] = "<<SCORE>> " + df["score"] + ", <<REPORT>> " + df["report"]

    return df


class PyTorchDataset(Dataset):
    def __init__(self, encodings, resultings):
        self.encodings = encodings
        self.resultings = resultings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.resultings["input_ids"][idx]
        return item

    def __len__(self):
        return len(self.resultings["input_ids"])


def transform_df_to_pytorch(df: pd.DataFrame, tokenizer: BartTokenizer) -> Dataset:
    inputs = list(df["res_and_job"])
    targets = list(df["score_and_report"])
    # Tokenize inputs and targets
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    dataset = PyTorchDataset(inputs, targets)

    return dataset


# for inference
def extract_score_and_report(text):
    pattern = r"(?i)SCORE\s+(\d+\.\d+)[,:]?[\s\n]+REPORT\s+(.*)"
    text = text.replace(">", "").replace("<", "")
    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        score = match.group(1)
        report = match.group(2)
        return float(score), report.strip()
    else:
        return None, None


def combine_data_and_generated_report(
    df: pd.DataFrame, generated_report: list
) -> pd.DataFrame:
    df["generated_report"] = generated_report
    df["score_from_report"], df["report_from_report"] = zip(
        *df["generated_report"].apply(extract_score_and_report)
    )
    return df
