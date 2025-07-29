import sys
import json
import re
import fitz
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import concurrent.futures
import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration is the same ---
MODEL_FILE = 'models/lgbm_filter.model'
CLASSES_FILE = 'models/lgbm_classes.npy'
TRANSFORMER_MODEL_DIR = 'models/transformer_specialist'
DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.75
FEATURE_COLUMNS = [
    'font_size', 'is_bold', 'is_italic', 'relative_font_size', 'x_position_normalized',
    'is_centered', 'space_below', 'text_length', 'is_all_caps', 'starts_with_numbering',
    'line_height', 'span_count'
]
# --- All helper functions (load_models, group_nearby_blocks, process_page_wrapper) are the same ---
def load_models():
    lgbm_model = lgb.Booster(model_file=MODEL_FILE)
    lgbm_classes = np.load(CLASSES_FILE, allow_pickle=True)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    classifier = pipeline("text-classification", model=transformer_model, tokenizer=tokenizer, device=DEVICE, return_all_scores=True)
    print("Hybrid models loaded successfully.")
    return lgbm_model, lgbm_classes, classifier

def group_nearby_blocks(blocks):
    if not blocks: return []
    merged_blocks, current_group = [], [blocks[0]]
    for next_block in blocks[1:]:
        last_block_in_group = current_group[-1]
        vertical_distance = next_block['bbox'][1] - last_block_in_group['bbox'][3]
        is_close_enough = vertical_distance < (last_block_in_group['lines'][0]['spans'][0]['size'] * 0.5)
        style_is_similar = abs(next_block['lines'][0]['spans'][0]['size'] - last_block_in_group['lines'][0]['spans'][0]['size']) < 1
        if is_close_enough and style_is_similar:
            current_group.append(next_block)
        else:
            if len(current_group) > 1:
                full_text = " ".join(" ".join(s['text'] for s in l['spans']) for b in current_group for l in b['lines']).strip()
                new_bbox = fitz.Rect(current_group[0]['bbox'])
                for b in current_group[1:]: new_bbox.include_rect(fitz.Rect(b['bbox']))
                base_block = current_group[0]
                base_block['bbox'] = list(new_bbox)
                base_block['lines'][0]['spans'][0]['text'] = full_text
                merged_blocks.append(base_block)
            else:
                merged_blocks.append(current_group[0])
            current_group = [next_block]
    if len(current_group) > 1:
        full_text = " ".join(" ".join(s['text'] for s in l['spans']) for b in current_group for l in b['lines']).strip()
        new_bbox = fitz.Rect(current_group[0]['bbox'])
        for b in current_group[1:]: new_bbox.include_rect(fitz.Rect(b['bbox']))
        base_block = current_group[0]
        base_block['bbox'] = list(new_bbox)
        base_block['lines'][0]['spans'][0]['text'] = full_text
        merged_blocks.append(base_block)
    else:
        merged_blocks.append(current_group[0])
    return merged_blocks

def process_page_wrapper(args):
    page_num, pdf_path_str, median_fs = args
    doc = fitz.open(pdf_path_str)
    page = doc[page_num]
    page_width = page.rect.width if page.rect.width > 0 else 1.0
    page_blocks = []
    numbering_re = re.compile(r'^\s*(\d+(\.\d+)*|[A-Za-z]\.|[IVXLCDM]+\.|•|■|-|[一-十]+[、．\.]|[（\(][一-十]+[）\)]|[①-⑳]|[０-９]+\.)\s+')
    initial_blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH)["blocks"]
    blocks = group_nearby_blocks([b for b in initial_blocks if b.get('type') == 0])
    for i, b in enumerate(blocks):
        try:
            spans = [s for l in b["lines"] for s in l["spans"]]
            if not spans: continue
            first_span = spans[0]
            block_text = " ".join(s["text"] for s in spans).strip()
        except (IndexError, KeyError): continue
        if not block_text: continue
        space_below = page.rect.height - b['bbox'][3]
        if i + 1 < len(blocks): space_below = blocks[i+1]['bbox'][1] - b['bbox'][3]
        center_x = (b['bbox'][0] + b['bbox'][2]) / 2
        is_centered = 1 if abs(center_x - page_width / 2) < (page_width * 0.15) else 0
        features = {
            "text": block_text, "page": page_num + 1, "y_pos": b['bbox'][1],
            "font_size": first_span.get('size', 12.0), "is_bold": 1 if "bold" in first_span.get('font', '').lower() else 0,
            "is_italic": 1 if "italic" in first_span.get('font', '').lower() else 0,
            "relative_font_size": first_span.get('size', 12.0) / median_fs if median_fs > 0 else 1,
            "x_position_normalized": b['bbox'][0] / page_width, "is_centered": is_centered,
            "space_below": space_below, "text_length": len(block_text),
            "is_all_caps": 1 if block_text.isupper() and len(block_text) > 1 else 0,
            "starts_with_numbering": 1 if numbering_re.match(block_text) else 0,
            "line_height": b['bbox'][3] - b['bbox'][1], "span_count": len(spans)
        }
        page_blocks.append(features)
    doc.close()
    return page_blocks

# --- Main Pipeline for Submission (Parallel) ---
def run_pipeline(pdf_path, lgbm_model, lgbm_classes, classifier):
    # This version uses parallel processing for speed in the final submission.
    try:
        doc = fitz.open(pdf_path)
        font_sizes = [s["size"] for page in doc for b in page.get_text("dict")["blocks"] if b.get('type')==0 for l in b["lines"] for s in l["spans"]]
        median_fs = np.median(font_sizes) if font_sizes else 12.0
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        return {"title": f"Error processing {pdf_path.name}", "outline": []}

    all_blocks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        tasks = [(i, str(pdf_path), median_fs) for i in range(num_pages)]
        results = executor.map(process_page_wrapper, tasks)
        for page_result in results: all_blocks.extend(page_result)

    if not all_blocks: return {"title": "", "outline": []}
    df = pd.DataFrame(all_blocks)
    df_for_prediction = df[FEATURE_COLUMNS]
    lgbm_preds = lgbm_model.predict(df_for_prediction)
    df['lgbm_label'] = [lgbm_classes[i] for i in np.argmax(lgbm_preds, axis=1)]
    candidates_df = df[df['lgbm_label'] != 'Body'].copy()
    if candidates_df.empty: return {"title": df.iloc[0]['text'][:100] if not df.empty else "", "outline": []}
    candidate_texts = candidates_df['text'].tolist()
    predictions = classifier(candidate_texts, batch_size=8)
    final_labels = []
    for i, pred_set in enumerate(predictions):
        lgbm_prediction = candidates_df['lgbm_label'].iloc[i]
        best_pred = max(pred_set, key=lambda x: x['score'])
        if best_pred['score'] >= CONFIDENCE_THRESHOLD:
            final_labels.append(best_pred['label'])
        else:
            final_labels.append(lgbm_prediction)
    candidates_df['final_label'] = final_labels
    doc_title = ""
    titles = candidates_df[candidates_df['final_label'] == 'Title']
    if not titles.empty:
        doc_title = titles.sort_values(by=['page', 'y_pos']).iloc[0]['text']
    else:
        doc_title = df.iloc[0]['text'][:100]
    headings = candidates_df[candidates_df['final_label'].isin(['H1', 'H2', 'H3', 'H4'])]
    outline = []
    for _, row in headings.sort_values(by=['page', 'y_pos']).iterrows():
        outline.append({"level": row['final_label'], "text": row['text'], "page": int(row['page'])})
    return {"title": doc_title, "outline": outline}


# --- NEW: Sequential Pipeline for Training ---
def run_pipeline_sequential(pdf_path, lgbm_model, lgbm_classes, classifier):
    # This version is identical but uses a simple for loop for stability during training.
    try:
        doc = fitz.open(pdf_path)
        font_sizes = [s["size"] for page in doc for b in page.get_text("dict")["blocks"] if b.get('type')==0 for l in b["lines"] for s in l["spans"]]
        median_fs = np.median(font_sizes) if font_sizes else 12.0
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        return {"title": f"Error processing {pdf_path.name}", "outline": []}

    all_blocks = []
    tasks = [(i, str(pdf_path), median_fs) for i in range(num_pages)]
    for task in tasks:
        page_result = process_page_wrapper(task)
        all_blocks.extend(page_result)

    if not all_blocks: return {"title": "", "outline": []}
    df = pd.DataFrame(all_blocks)
    df_for_prediction = df[FEATURE_COLUMNS]
    lgbm_preds = lgbm_model.predict(df_for_prediction)
    df['lgbm_label'] = [lgbm_classes[i] for i in np.argmax(lgbm_preds, axis=1)]
    candidates_df = df[df['lgbm_label'] != 'Body'].copy()
    if candidates_df.empty: return {"title": df.iloc[0]['text'][:100] if not df.empty else "", "outline": []}
    candidate_texts = candidates_df['text'].tolist()
    predictions = classifier(candidate_texts, batch_size=8)
    final_labels = []
    for i, pred_set in enumerate(predictions):
        lgbm_prediction = candidates_df['lgbm_label'].iloc[i]
        best_pred = max(pred_set, key=lambda x: x['score'])
        if best_pred['score'] >= CONFIDENCE_THRESHOLD:
            final_labels.append(best_pred['label'])
        else:
            final_labels.append(lgbm_prediction)
    candidates_df['final_label'] = final_labels
    doc_title = ""
    titles = candidates_df[candidates_df['final_label'] == 'Title']
    if not titles.empty:
        doc_title = titles.sort_values(by=['page', 'y_pos']).iloc[0]['text']
    else:
        doc_title = df.iloc[0]['text'][:100]
    headings = candidates_df[candidates_df['final_label'].isin(['H1', 'H2', 'H3', 'H4'])]
    outline = []
    for _, row in headings.sort_values(by=['page', 'y_pos']).iterrows():
        outline.append({"level": row['final_label'], "text": row['text'], "page": int(row['page'])})
    return {"title": doc_title, "outline": outline}


def main():
    # This main block is for the final submission. It uses the parallel pipeline.
    if __name__ == '__main__':
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(input_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found in /app/input.")
            exit()

        lgbm_model, lgbm_classes, classifier = load_models()
        
        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            result = run_pipeline_sequential(pdf_file, lgbm_model, lgbm_classes, classifier)
            
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(result, f, indent=4)
            
            print(f"✅ Finished processing {pdf_file.name} -> {output_file.name}")

if __name__ == '__main__':
    main()