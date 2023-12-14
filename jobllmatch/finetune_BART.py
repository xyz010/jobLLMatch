import argparse
import logging
import pickle
import sys
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from data_preprocess import (
    create_dataset_prompt_col,
    create_dataset_splits,
    data_preprocess,
    transform_df_to_pytorch,
)
from torch.utils.data import DataLoader
from transformers import AdamW, BartForConditionalGeneration, BartTokenizer

sys.path.append(".")
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, required=True)
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--lr", type=float, default=1e-5, required=False)
    parser.add_argument("--eps", type=float, default=1e-8, required=False)
    parser.add_argument("--patience", type=int, default=3, required=False)
    parser.add_argument("--test_mode", action="store_true", default=False)
    parser.add_argument("--test_mode_size", type=int, default=10)
    parser.add_argument("--validation_freq", type=int, default=10, required=False)
    parser.add_argument("--datetime", type=str, required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    datetime = args.datetime
    logging.info(f"datetime: {datetime}")
    logging.info("Reading data")
    with open("res_job_score_report_combined.pkl", "rb") as f:
        df = pickle.load(f)
    logging.info("Data read")
    logging.info(f"Shape of the data is {df.shape}")

    df_processed = data_preprocess(df)
    df_processed = create_dataset_prompt_col(df_processed)

    # train 70%, validation 15%, test 15%
    df_train, df_val, df_test = create_dataset_splits(
        df_processed, train_split=0.3, val_split=0.5, random_state=42
    )

    if args.test_mode:
        logging.info("Testing setup. Only using test_mode_size samples")
        df_train = df_train[: args.test_mode_size]
        df_val = df_val[: args.test_mode_size]
        df_test = df_test[: args.test_mode_size]
        logging.info(f"Shape of the training data is {df_train.shape}")

    logging.info("Initializing model and tokenizer")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model.to("cuda")
    logging.info("Model initialized and sent to device")

    # transform data into pytorch format
    dataset_train = transform_df_to_pytorch(df_train, tokenizer)
    dataset_val = transform_df_to_pytorch(df_val, tokenizer)
    # dataset_test = transform_df_to_pytorch(df_test, tokenizer)

    batch_size = args.batch_size
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    epochs = args.epochs
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.eps)
    training_losses = []
    validation_losses = []
    best_val_loss = float("inf")
    patience = args.patience

    logging.info(
        f"len(loader_train): {len(loader_train)}, args.batch_size: {args.batch_size}, args.validation_freq: {args.validation_freq}"
    )
    validation_freq = len(loader_train) // args.validation_freq
    logging.info(
        f"Validation frequency: {args.validation_freq}, every {validation_freq} batch"
    )
    # INFO:root:Validation frequency: 10, every 1 step
    logging.info("Starting training...")
    for epoch in range(epochs):
        logging.info(f"Epoch: {epoch}")

        # Training Step
        model.train()
        total_train_loss = 0
        total_val_loss = 0
        val_loss_count = (
            0  # Variable to count the number of times validation loss is calculated
        )
        for step, batch in enumerate(loader_train):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to("cuda")
            attention_mask = batch["attention_mask"].to("cuda")
            resultings = batch["labels"].to("cuda")

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=resultings
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # if step is a multiple of validation_freq, then enter
            if step % validation_freq == 0 and step != 0:
                model.eval()
                logging.info(f"Inside validation, epoch: {epoch}, step: {step}")
                with torch.no_grad():
                    for val_batch in loader_val:
                        val_input_ids = val_batch["input_ids"].to("cuda")
                        val_attention_mask = val_batch["attention_mask"].to("cuda")
                        val_labels = val_batch["labels"].to("cuda")

                        val_outputs = model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_labels,
                        )
                        val_loss = val_outputs.loss

                        total_val_loss += val_loss.item()
                        val_loss_count += 1
                    # logging.info(f"total_val_loss: {total_val_loss}, epoch: {epoch}, step: {step}")

        # Calcuate Training loss
        avg_train_loss = total_train_loss / len(loader_train)
        training_losses.append(avg_train_loss)
        logging.info(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")

        # Calculate Validation loss
        # avg_val_loss = total_val_loss / len(loader_val)
        avg_val_loss = total_val_loss / val_loss_count if val_loss_count > 0 else 0
        validation_losses.append(avg_val_loss)
        logging.info(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

        #         # Validation Step
        #         model.eval()
        #         total_val_loss = 0
        #         with torch.no_grad():
        #             for batch in loader_val:
        #                 input_ids = batch["input_ids"].to("cuda")
        #                 attention_mask = batch["attention_mask"].to("cuda")
        #                 labels = batch["labels"].to("cuda")

        #                 outputs = model(
        #                     input_ids=input_ids, attention_mask=attention_mask, labels=labels
        #                 )
        #                 loss = outputs.loss

        #                 total_val_loss += loss.item()
        #         avg_val_loss = total_val_loss / len(loader_val)
        #         validation_losses.append(avg_val_loss)
        #         logging.info(f"Epoch: {epoch}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info("Early stopping triggered. Stopping training.")
            break

        model.train()

        plt.figure(figsize=(10, 5))
        plt.title("Training and Validation Loss")
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(
            f"plots/fine_tune_BART_after_mid_{epochs}_lr_{args.lr}_eps_{args.eps}.png"
        )
        plt.show()

        if args.test_mode:
            mode = "test"
        else:
            mode = "prod"

        if args.test_mode:
            datetime += "_test"
        output_path = f"model_ckpt/{datetime}/epoch_{epoch}_lr_{args.lr}_eps_{args.eps}_mode_{mode}"
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        df_results = pd.DataFrame(
            {"training_loss": training_losses, "validation_loss": validation_losses}
        )
        df_results.to_pickle(output_path + "/losses.pkl")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time()
    runtime_minutes = (elapsed - start) / 60
    if runtime_minutes > 60:
        runtime_hours = runtime_minutes / 60
        logging.info(f"Runtime of the program is {runtime_hours} hours")
    else:
        logging.info(f"Runtime of the program is {runtime_minutes} minutes")
        logging.info(f"Runtime of the program is {runtime_minutes} minutes")
