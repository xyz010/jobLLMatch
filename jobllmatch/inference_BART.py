import argparse
import logging
import pickle
import time

import torch
from data_preprocess import (
    combine_data_and_generated_report,
    create_dataset_prompt_col,
    create_dataset_splits,
    data_preprocess,
    transform_df_to_pytorch,
)
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer


def main():
    """Generates Scores and Reports using the fine-tuned BART model on the test set"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="checkpoint path")
    parser.add_argument("--test_mode", action="store_true", default=False)
    parser.add_argument("--test_mode_size", type=int, default=10)
    parser.add_argument("--csv", action="store_true", default=False)
    parser.add_argument("--calculate_loss", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Reading data")
    with open("res_job_score_report_combined.pkl", "rb") as f:
        df = pickle.load(f)
    logging.info("Data read")

    df_processed = data_preprocess(df)
    df_processed = create_dataset_prompt_col(df_processed)

    # train 70%, validation 15%, test 15%
    _, _, df_test = create_dataset_splits(
        df_processed, train_split=0.3, val_split=0.5, random_state=42
    )

    if args.test_mode:
        logging.info(f"Testing setup. Only using {args.test_mode_size} samples")
        df_test = df_test[: args.test_mode_size]
    logging.info(f"Shape of the test data is {df_test.shape}")

    if not args.checkpoint:
        logging.info("Loading model from bart-large")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    else:
        checkpoint = args.checkpoint
        logging.info("Initializing model and tokenizer from checkpoint")
        tokenizer = BartTokenizer.from_pretrained(checkpoint)
        model = BartForConditionalGeneration.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        model.to(device)
    logging.info(f"Model initialized and sent to device: {device}")

    dataset_test = transform_df_to_pytorch(df_test, tokenizer)
    loader_test = DataLoader(dataset_test, batch_size=3, shuffle=False)

    model.eval()
    total_test_loss = 0
    res = []
    with torch.no_grad():
        for test_batch in loader_test:
            test_input_ids = test_batch["input_ids"].to("cuda")
            test_attention_mask = test_batch["attention_mask"].to("cuda")
            test_labels = test_batch["labels"].to("cuda")

            report_ids = model.generate(
                input_ids=test_input_ids,
                attention_mask=test_attention_mask,
                max_length=256,
                num_return_sequences=1,
            )
            decoded_outputs = tokenizer.batch_decode(
                report_ids, skip_special_tokens=True
            )
            res.extend(decoded_outputs)

            if args.calculate_loss:
                test_outputs = model(
                    input_ids=test_input_ids,
                    attention_mask=test_attention_mask,
                    labels=test_labels,
                )
                test_loss = test_outputs.loss
                total_test_loss += test_loss.item()
    if args.calculate_loss:
        logging.info(f"Total validation loss is {total_test_loss/len(loader_test)}")

    # now combine the genrated report with the score
    combined_df_test_and_results = combine_data_and_generated_report(df_test, res)
    out_dir = "combined_data/"

    if args.csv:
        filename = f"{out_dir}combined_df_test_and_results.csv"
        combined_df_test_and_results.to_csv(filename)
    else:
        filename = f"{out_dir}combined_df_test_and_results.pkl"
        combined_df_test_and_results.to_pickle(filename)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f"Time taken: {end-start} seconds")
