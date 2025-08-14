from datasets import load_dataset

def preprocess_data(dataset_name, tokenizer, max_length=512):
    """
    Preprocess the dataset by tokenizing and truncating/padding.

    Args:
        dataset_name (str): Name of the dataset.
        tokenizer: Tokenizer to be used for tokenization.
        max_length (int): Maximum length of the tokenized sequences.

    Returns:
        dict: Processed train and validation datasets.
    """
    dataset = load_dataset(dataset_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets["train"], tokenized_datasets["validation"]
