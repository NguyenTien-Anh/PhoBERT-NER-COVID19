# Vietnamese Named Entity Recognition (NER) with phoBERT and FastAPI

## Description

This project provides a backend API for Named Entity Recognition (NER) in Vietnamese text. It utilizes a fine-tuned `phoBERT` model and is served using a high-performance FastAPI backend.

## Demo

![API Demo](./demo.gif)

## Key Features

- **State-of-the-art Model:** NER model based on `phoBERT` fine-tuned for Vietnamese.
- **High-Performance Backend:** Built with FastAPI for fast and reliable performance.
- **`/predict` Endpoint:** Receives text and returns recognized entities with their labels and confidence scores.
- **Command-Line Interface:** Includes `infer.py` for direct predictions from the terminal.
- **Custom Vietnamese Tokenizer:** Integrates a custom tokenizer for effective text processing.

## Installation

Follow these steps to set up and run the project locally.

**1. Prerequisites:**
- Python 3.10+

**2. Clone the Repository:**
```bash
git clone <repository-url>
cd phoBERT-ner-covid
```

**3. Create and Activate Virtual Environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**4. Install Dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the API Backend

To start the API server, run the following command. The `MODEL_PATH` environment variable should point to the directory containing your fine-tuned model checkpoint.

```bash
MODEL_PATH=test-ner/checkpoint-1000 .venv/bin/uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

### 2. Call the API

Use `curl` or any API client to send a `POST` request to the `/predict` endpoint.

```bash
curl -X 'POST' 'http://127.0.0.1:8000/predict' \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
  "text": "Bác sĩ Nguyễn Văn A đang làm việc tại Bệnh viện Chợ Rẫy."
}'
```

### 3. Use the Command-Line Interface (CLI)

You can also perform predictions directly from the terminal using `infer.py`.

```bash
python infer.py --model_path test-ner/checkpoint-1000 --text "Bác sĩ Nguyễn Văn A đang làm việc tại Bệnh viện Chợ Rẫy."
```

## API Structure

### Request Body

The `/predict` endpoint expects a JSON object with a `text` field.

```json
{
  "text": "Đây là câu văn bản tiếng Việt cần dự đoán."
}
```

### Response Body

The API returns a JSON object containing the tokens, predicted tags, and confidence scores.

```json
{
  "tokens": ["Bác", "sĩ", "Nguyễn", "Văn", "A", "đang", "làm", "việc", "tại", "Bệnh", "viện", "Chợ", "Rẫy", "."],
  "tags": ["O", "O", "B-NAME", "B-NAME", "B-NAME", "O", "O", "O", "O", "B-LOCATION", "I-LOCATION", "I-LOCATION", "I-LOCATION", "O"],
  "confidence_scores": [0.9888, 0.9706, 0.5547, 0.7155, 0.8539, 0.9937, 0.9936, 0.9935, 0.9936, 0.9731, 0.9774, 0.9780, 0.9773, 0.9937]
}
```

## Technologies Used

- **Backend:** FastAPI, Uvicorn
- **ML/DL:** PyTorch, Hugging Face Transformers
- **Data Validation:** Pydantic

