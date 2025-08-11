import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

print("--- Training Hybrid AI Models from CSV Dataset ---")

# --- Configuration ---
CSV_PATH = Path("data/labeled_blocks.csv")
LGBM_MODEL_SAVE_PATH = 'models/lgbm_filter.model'
TRANSFORMER_MODEL_SAVE_PATH = 'models/transformer_specialist'
CLASSES_SAVE_PATH = 'models/lgbm_classes.npy'
os.makedirs(os.path.dirname(LGBM_MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(TRANSFORMER_MODEL_SAVE_PATH, exist_ok=True)

FEATURE_COLUMNS = [
    'font_size', 'is_bold', 'is_italic', 'relative_font_size', 'x_position_normalized',
    'is_centered', 'space_below', 'text_length', 'is_all_caps', 'starts_with_numbering',
    'line_height', 'span_count'
]

# --- Transformer Dataset Class ---
class HeadingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

# Add to training loop
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

# --- Main Training ---
try:
    training_df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"❌ Error: {CSV_PATH} not found. Please create this file and add your training data.")
    exit()

if training_df is not None and not training_df.empty:
    # --- Train LightGBM Filter ---
    print("\n--- Training LightGBM Filter ---")
    le_lgbm = LabelEncoder()
    training_df['label_encoded'] = le_lgbm.fit_transform(training_df['label'])
    
    for col in FEATURE_COLUMNS:
        training_df[col] = pd.to_numeric(training_df[col], errors='coerce')
    training_df = training_df.dropna(subset=FEATURE_COLUMNS)

    X = training_df[FEATURE_COLUMNS]
    y = training_df['label_encoded']
    
    lgb_clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(le_lgbm.classes_), random_state=42)
    lgb_clf.fit(X, y)
    lgb_clf.booster_.save_model(LGBM_MODEL_SAVE_PATH)
    np.save(CLASSES_SAVE_PATH, le_lgbm.classes_)
    print(f"✅ LightGBM filter saved to {LGBM_MODEL_SAVE_PATH}")

    # --- Train Transformer Specialist ---
    print("\n--- Fine-Tuning Language Model Specialist ---")
    headings_df = training_df[training_df['label'] != 'Body'].copy()
    
    if headings_df.empty or len(headings_df) < 10:
        print("⚠️ Warning: Not enough heading/title samples to train the language model. Skipping.")
    else:
        le_transformer = LabelEncoder()
        headings_df['label_encoded'] = le_transformer.fit_transform(headings_df['label'])
        
        MODEL_NAME = 'prajjwal1/bert-tiny'
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(le_transformer.classes_),
            id2label={i: label for i, label in enumerate(le_transformer.classes_)},
            label2id={label: i for i, label in enumerate(le_transformer.classes_)}
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            headings_df['text'].tolist(), headings_df['label_encoded'].tolist(), 
            test_size=0.2, random_state=42, stratify=headings_df['label_encoded']
        )
        train_dataset = HeadingDataset(train_texts, train_labels, tokenizer)
        val_dataset = HeadingDataset(val_texts, val_labels, tokenizer)

        training_args = TrainingArguments(
            output_dir='./training_results', num_train_epochs=200, per_device_train_batch_size=8,
            learning_rate=5e-5, weight_decay=0.01, eval_strategy="epoch",
            save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="loss", greater_is_better=False,
            save_total_limit=1
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics  # Add this line
        )
        trainer.train()
        trainer.save_model(TRANSFORMER_MODEL_SAVE_PATH)
        tokenizer.save_pretrained(TRANSFORMER_MODEL_SAVE_PATH)
        print(f"✅ Language model specialist saved to {TRANSFORMER_MODEL_SAVE_PATH}")
else:
    print("--- Training skipped. CSV file is empty or invalid. ---")