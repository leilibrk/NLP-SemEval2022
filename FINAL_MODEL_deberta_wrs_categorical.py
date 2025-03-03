import gc
import torch
import pandas as pd
import ast  # for list-like labels, if needed
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    DebertaTokenizer,
    DebertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from evaluate import load
import numpy as np
import optuna  # Using optuna for hyperparameter optimization

# Clean up GPU memory
gc.collect()
torch.cuda.empty_cache()

import re

# -------------------------------
# Data Preparation and Tokenization
# -------------------------------

# Define column names and load datasets
columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label']
df_texts = pd.read_csv("dontpatronizeme_pcl.tsv", sep="\t", header=None, names=columns)
df_labels = pd.read_csv("train_semeval_parids-labels.csv")

# Ensure the IDs are strings for merging
df_texts["par_id"] = df_texts["par_id"].astype(str)
df_labels["par_id"] = df_labels["par_id"].astype(str)

# Drop unnecessary label column from df_labels and create binary labels in df_texts
df_labels = df_labels.drop(columns=["label"])
df_texts["binary_label"] = df_texts["label"].apply(lambda x: 1 if x >= 2 else 0)
df_texts = df_texts.drop(columns=["label"])

# Merge datasets on paragraph ID and rename the binary label to "label"
df = df_labels.merge(df_texts, on="par_id", how="left")
df.rename(columns={"binary_label": "label"}, inplace=True)
df = df.dropna(subset=["text", "label"])  # Drop rows with missing data


def clean_text(text):
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
   
    return text.strip()
df["text"] = df["text"].astype(str).apply(clean_text)
print("Preprocessing and balancing complete!")

# Initialize the tokenizer and model
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
model = DebertaForSequenceClassification.from_pretrained(
    "microsoft/deberta-base", num_labels=2, ignore_mismatched_sizes=True
)

def create_combined_text(keyword, country_code, text):
    # You can format the string as you like. For example, using delimiters helps the model to distinguish the parts.
    return f"keyword: {keyword} | country: {country_code} | text: {text}"

# Tokenization function
def tokenize_function(text):
    return tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

# Tokenize the texts and stack tensors
# df["tokenized"] = df["text"].apply(lambda x: tokenize_function(x))
df["combined_text"] = df.apply(lambda row: create_combined_text(row["keyword"], row["country_code"], row["text"]), axis=1)
df["tokenized"] = df["combined_text"].apply(lambda x: tokenize_function(x))

input_ids = torch.cat([t["input_ids"] for t in df["tokenized"]], dim=0)
attention_masks = torch.cat([t["attention_mask"] for t in df["tokenized"]], dim=0)
labels = torch.tensor(df["label"].values, dtype=torch.long)

# -------------------------------
# Create Dataset and Sampler
# -------------------------------

class PCLDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }

# Split data into training and validation sets using indices
indices = list(range(len(df)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_ids, val_ids = input_ids[train_idx], input_ids[val_idx]
train_masks, val_masks = attention_masks[train_idx], attention_masks[val_idx]
train_labels, val_labels = labels[train_idx], labels[val_idx]

train_dataset = PCLDataset(train_ids, train_masks, train_labels)
val_dataset = PCLDataset(val_ids, val_masks, val_labels)

# -------------------------------
# Define a Custom Trainer with Weighted Sampling
# -------------------------------

class WeightedSamplerTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # Compute class counts from the dataset labels
        labels_tensor = torch.tensor(self.train_dataset.labels)
        class_counts = torch.bincount(labels_tensor)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels_tensor.numpy()]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(self.train_dataset), replacement=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
        )

# -------------------------------
# Define Metric Computation
# -------------------------------

def compute_metrics(eval_pred):
    """Computes Accuracy and F1 Score"""
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    references = eval_pred.label_ids
    accuracy_score = accuracy_metric.compute(predictions=predictions, references=references)
    f1_score = f1_metric.compute(predictions=predictions, references=references)
    return {"accuracy": accuracy_score["accuracy"], "f1": f1_score["f1"]}

# -------------------------------
# Final Training with Best Hyperparameters and Early Stopping
# -------------------------------

# best_params = study.best_trial.params
final_training_args = TrainingArguments(
       output_dir="./results",
       learning_rate=9.468961305919929e-06,
       optim="adamw_torch",
       warmup_ratio=0.1,
       evaluation_strategy="epoch",
       save_strategy="epoch", # change to no
       per_device_train_batch_size=4,  # Reduce from 8 or 16 to 4 or even 2
       per_device_eval_batch_size=4,   # Match train batch size
       num_train_epochs=6,
       weight_decay=0.03430359963348227,
       logging_dir="./logs",
       logging_steps=50,
       save_total_limit=2,
       lr_scheduler_type="linear",
       load_best_model_at_end=True,
       metric_for_best_model="f1",
       report_to="none",
       fp16=False,  # Enables mixed precision training (reduces memory usage)
       bf16=False,  # Keep this False unless on Ampere GPUs (A100, RTX 30xx)
   )

final_trainer = WeightedSamplerTrainer(
    model=model,
    args=final_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Adjust the patience as needed

)

final_trainer.train()
final_results = final_trainer.evaluate()
print("Final evaluation results:", final_results)

# Save the final model and tokenizer
model.save_pretrained("./deberta__keyword_WS_model")
tokenizer.save_pretrained("./deberta__keyword_WS_model")

