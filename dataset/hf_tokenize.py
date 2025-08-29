from datasets import load_metric
import datasets
from transformers import AutoTokenizer
from dataset.covid19_dataset import COVID19Dataset

metric = load_metric("seqeval")
logger = datasets.logging.get_logger(__name__)


class HFTokenizer(object):
    NAME = "HFTokenizer"

    def __init__(self,
                 hf_pretrained_tokenizer_checkpoint):
        try:
            # Try to load fast tokenizer (Rust-based) that supports word_ids()
            self._tokenizer = AutoTokenizer.from_pretrained(
                hf_pretrained_tokenizer_checkpoint, use_fast=True)
        except Exception as e:
            logger.warning(f"Fast tokenizer not available, falling back to slow tokenizer: {e}")
            # Fallback to slow tokenizer (Python-based)
            self._tokenizer = AutoTokenizer.from_pretrained(
                hf_pretrained_tokenizer_checkpoint, use_fast=False)

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def init_vf(hf_pretrained_tokenizer_checkpoint):
        return HFTokenizer(hf_pretrained_tokenizer_checkpoint)

    def tokenize_and_align_labels(self,
                                  examples,
                                  label_all_tokens=False):
        tokenized_inputs = self._tokenizer(examples["tokens"],
                                           truncation=True,
                                           is_split_into_words=True,)

        labels = []
        prediction_masks = []

        for i, label in enumerate(examples[f"ner_tags"]):
            if self._tokenizer.is_fast:
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids, prediction_mask = self._align_labels_with_word_ids(
                    word_ids, label, label_all_tokens)
            else:
                label_ids, prediction_mask = self._align_labels_without_word_ids(
                    examples["tokens"][i], tokenized_inputs["input_ids"][i],
                    label, label_all_tokens)

            assert len(label_ids) == len(prediction_mask)
            labels.append(label_ids)
            prediction_masks.append(prediction_mask)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["prediction_mask"] = prediction_masks

        return tokenized_inputs

    def _align_labels_with_word_ids(self, word_ids, label, label_all_tokens):
        """Align labels using word_ids() method (for fast tokenizers)"""
        previous_word_idx = None
        label_ids = []
        prediction_mask = []

        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
                prediction_mask.append(False)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                prediction_mask.append(True)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label[word_idx] if label_all_tokens else -100)
                prediction_mask.append(False)
            previous_word_idx = word_idx

        return label_ids, prediction_mask

    def _align_labels_without_word_ids(self, tokens, input_ids, label, label_all_tokens):
        """Align labels without word_ids() method (for slow tokenizers)"""
        label_ids = []
        prediction_mask = []

        # Get special token ids
        cls_token_id = self._tokenizer.cls_token_id
        sep_token_id = self._tokenizer.sep_token_id
        pad_token_id = self._tokenizer.pad_token_id

        # Track current word index
        word_idx = 0
        i = 0

        while i < len(input_ids):
            token_id = input_ids[i]

            # Handle special tokens
            if token_id in [cls_token_id, sep_token_id, pad_token_id]:
                label_ids.append(-100)
                prediction_mask.append(False)
                i += 1
                continue

            # Check if we've processed all words
            if word_idx >= len(tokens):
                label_ids.append(-100)
                prediction_mask.append(False)
                i += 1
                continue

            # Tokenize the current word to see how many subword tokens it produces
            word_tokens = self._tokenizer.tokenize(tokens[word_idx])

            # Assign labels to subword tokens
            for j, _ in enumerate(word_tokens):
                if i < len(input_ids):
                    if j == 0:  # First subword token gets the label
                        label_ids.append(label[word_idx])
                        prediction_mask.append(True)
                    else:  # Subsequent subword tokens
                        label_ids.append(
                            label[word_idx] if label_all_tokens else -100)
                        prediction_mask.append(False)
                    i += 1

            word_idx += 1

        return label_ids, prediction_mask


if __name__ == "__main__":

    hf_pretrained_tokenizer_checkpoint = "vinai/phobert-base"
    dataset = COVID19Dataset().dataset

    hf_preprocessor = HFTokenizer.init_vf(
        hf_pretrained_tokenizer_checkpoint=hf_pretrained_tokenizer_checkpoint)

    tokenized_datasets = dataset.map(
        hf_preprocessor.tokenize_and_align_labels, batched=True)

    print(dataset)

    print("*" * 100)

    print(tokenized_datasets)

    print("First sample: ", dataset['train'][0])

    print("*" * 100)

    print("First tokenized sample: ", tokenized_datasets['train'][0])
