import argparse
import torch
from transformers import AutoTokenizer, RobertaForTokenClassification
import re
from typing import List

def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenize Vietnamese text for NER processing.
    This version handles complex patterns like date ranges and attached punctuation.

    Args:
        text: Input Vietnamese text

    Returns:
        List of tokens
    """
    if not text.strip():
        return []

    # Replace underscores, then use regex for tokenization
    text = text.strip().replace('_', ' ')

    # Regex to find words (sequences of word characters) or any single non-whitespace, non-word character.
    # This effectively separates words, numbers, and punctuation.
    # For example, "24-7," becomes ["24", "-", "7", ","]
    tokens = re.findall(r'\d+/\d+|(?:[A-Z]\.)+[A-Z]?|\w+|[^\w\s]', text)

    return [token for token in tokens if token.strip()]

def predict_ner(model_path, text, tokenizer, model):
    """
    Thực hiện dự đoán NER trên một câu văn bản.

    Args:
        model_path (str): Đường dẫn đến thư mục chứa mô hình.
        text (str): Câu văn bản tiếng Việt cần dự đoán.
        tokenizer: Tokenizer đã được tải.
        model: Mô hình đã được tải.

    Returns:
        Tuple[List[str], List[str], List[float]]: Một tuple chứa danh sách các token,
        nhãn dự đoán và điểm tin cậy tương ứng.
    """
    id2label = model.config.id2label

    # Sử dụng tokenizer tiếng Việt tùy chỉnh
    tokens = tokenize_vietnamese(text)
    if not tokens:
        return [], [], []

    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    probabilities = torch.softmax(logits, dim=2)
    predictions = torch.argmax(probabilities, dim=2)
    confidence_scores = torch.max(probabilities, dim=2).values

    predicted_token_labels = [id2label[t.item()] for t in predictions[0]]

    aligned_tokens = []
    aligned_tags = []
    aligned_scores = []

    if tokenizer.is_fast:
        word_ids = inputs.word_ids()
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == previous_word_idx:
                continue

            original_word = tokens[word_idx]
            label = predicted_token_labels[i]
            score = confidence_scores[0][i].item()

            aligned_tokens.append(original_word)
            aligned_tags.append(label)
            aligned_scores.append(score)

            previous_word_idx = word_idx
    else:  # Fallback cho tokenizer chậm
        original_token_idx = 0
        for i in range(1, len(predicted_token_labels) - 1):
            if original_token_idx < len(tokens):
                label = predicted_token_labels[i]
                score = confidence_scores[0][i].item()
                aligned_tokens.append(tokens[original_token_idx])
                aligned_tags.append(label)
                aligned_scores.append(score)
                original_token_idx += 1

    return aligned_tokens, aligned_tags, aligned_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán NER cho văn bản tiếng Việt bằng mô hình đã huấn luyện.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Đường dẫn đến thư mục chứa mô hình và tokenizer đã được fine-tune."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Câu văn bản tiếng Việt cần dự đoán."
    )

    args = parser.parse_args()

    try:
        loaded_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        loaded_model = RobertaForTokenClassification.from_pretrained(args.model_path)
        print(f"Tải mô hình và tokenizer từ '{args.model_path}' thành công.")
    except Exception as e:
        print(f"Lỗi: Không thể tải mô hình hoặc tokenizer từ '{args.model_path}'. Vui lòng kiểm tra lại đường dẫn.")
        print(e)
        exit()

    tokens, tags, scores = predict_ner(args.model_path, args.text, loaded_tokenizer, loaded_model)

    print("\n--- Kết quả dự đoán NER ---")
    for token, tag, score in zip(tokens, tags, scores):
        print(f"{token}: {tag} (Confidence: {score:.4f})")

