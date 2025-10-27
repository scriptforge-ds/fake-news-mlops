#!/usr/bin/env python3
"""
Preprocessing script for BERT-based pipeline.

Creates:
 - data/processed/train.csv
 - data/processed/val.csv
 - data/processed/test.csv
 - data/processed/tokenized/  (Hugging Face dataset saved to disk)

Usage (example):
    python src/data/preprocess.py \
        --input_csv data/processed/cleaned_for_preprocessing.csv \
        --out_dir data/processed \
        --model_name distilbert-base-uncased \
        --max_len 256 \
        --test_size 0.2 \
        --val_ratio 0.5 \
        --random_state 42
"""
import os
import re
import html
import argparse
import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def clean_minimal(text: str) -> str:
    """Minimal cleaning appropriate for transformers:
       - unescape HTML entities
       - remove URLs
       - remove HTML tags
       - normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    # remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_input(input_csv: str) -> pd.DataFrame:
    """Load the cleaned CSV produced by EDA stage."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info("Loaded input CSV with shape %s", df.shape)
    return df


def prepare_combined_text(df: pd.DataFrame, title_col: str = "title", text_col: str = "text") -> pd.DataFrame:
    """Create a combined_text column from title + text and perform minimal cleaning."""
    # Ensure columns exist
    for col in (title_col, text_col):
        if col not in df.columns:
            df[col] = ""

    # combine and clean
    combined = (df[title_col].fillna("").astype(str) + " " + df[text_col].fillna("").astype(str))
    tqdm.pandas(desc="Cleaning texts")
    df["combined_text"] = combined.progress_apply(clean_minimal)
    # Keep only relevant columns
    if "label" not in df.columns:
        raise KeyError("Input dataframe must contain a 'label' column (0 or 1).")
    # Ensure label is integer
    df["label"] = df["label"].astype(int)
    return df[["combined_text", "label"]]


def stratified_splits(df: pd.DataFrame, test_size: float = 0.2, val_ratio: float = 0.5, random_state: int = 42
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test, stratified by label.
    - test_size: proportion of total to hold out as temp (val+test)
    - val_ratio: proportion of temp to assign to validation (rest becomes test)
    """
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=random_state)
    val_df, test_df = train_test_split(temp_df, test_size=(1 - val_ratio), stratify=temp_df["label"], random_state=random_state)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenize_and_save(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      test_df: pd.DataFrame,
                      model_name: str,
                      max_length: int,
                      out_dir: str) -> None:
    """
    Tokenize datasets using specified transformer tokenizer and save a DatasetDict to disk.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    logger.info("Loaded tokenizer: %s", model_name)

    # Convert pandas -> HF Datasets
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    # Ensure label column name aligns with Trainer expectations
    # rename 'label' -> 'labels' if necessary
    def ensure_labels(dataset):
        if "label" in dataset.column_names and "labels" not in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        return dataset

    train_ds = ensure_labels(train_ds)
    val_ds = ensure_labels(val_ds)
    test_ds = ensure_labels(test_ds)

    dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

    def tokenize_batch(batch):
        return tokenizer(batch["combined_text"], padding="max_length", truncation=True, max_length=max_length)

    logger.info("Tokenizing datasets (this may take a few minutes)...")
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["combined_text"], desc="Tokenizing")
    # Ensure format is torch-ready later; we just save to disk
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "tokenized")
    tokenized.save_to_disk(save_path)
    logger.info("Tokenized datasets saved to: %s", save_path)


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str) -> None:
    """Save CSV splits to disk."""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved splits to %s", out_dir)
    logger.info("Train: %d rows, Val: %d rows, Test: %d rows", len(train_df), len(val_df), len(test_df))


def main(args):
    # Load input
    df_raw = load_input(args.input_csv)

    # Prepare combined text and minimal cleaning
    df = prepare_combined_text(df_raw, title_col=args.title_col, text_col=args.text_col)

    # Splits
    train_df, val_df, test_df = stratified_splits(df, test_size=args.test_size, val_ratio=args.val_ratio,
                                                  random_state=args.random_state)

    # Save CSVs
    save_splits(train_df, val_df, test_df, args.out_dir)

    # Tokenize and save tokenized dataset
    tokenize_and_save(train_df, val_df, test_df, model_name=args.model_name, max_length=args.max_len, out_dir=args.out_dir)

    # Quick sanity prints
    logger.info("Sanity checks:")
    for name, d in zip(["train", "validation", "test"], [train_df, val_df, test_df]):
        # label distribution
        total = len(d)
        n_fake = int(d["label"].sum())
        n_real = total - n_fake
        logger.info("%s -> total=%d, fake=%d, real=%d", name, total, n_fake, n_real)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset for BERT-based training.")
    parser.add_argument("--input_csv", type=str, default="data/processed/cleaned_for_preprocessing.csv",
                        help="Input cleaned CSV from EDA stage.")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                        help="Directory to save train/val/test and tokenized dataset.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Hugging Face tokenizer/model name.")
    parser.add_argument("--max_len", type=int, default=256, help="Max token length for tokenizer.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion reserved for temp (val+test).")
    parser.add_argument("--val_ratio", type=float, default=0.5,
                        help="Proportion of temp set used for validation. (val_ratio=0.5 => val=test)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for splits.")
    parser.add_argument("--title_col", type=str, default="title", help="Column name for title.")
    parser.add_argument("--text_col", type=str, default="text", help="Column name for article body.")
    args = parser.parse_args()
    main(args)