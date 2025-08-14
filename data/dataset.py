import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import os

class InstructionDataset(Dataset):
    def __init__(self, dataset_name, model_name, split="train", max_length=1024):
        # Load JSON/CSV dataset with fields `cot` (reasoning chain) & `answer`
        print('before loading the dataset')
        ds = load_dataset(dataset_name, split=split)
        print('after loading the dataset')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            #self.tokenizer.pad_token = self.tokenizer.eos_token  # Option 1
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Option 2

        # if self._try_load_cached():
        #     print(f"✅ Loaded cached {split} dataset from {self.cache_path}")
        # else:
        # Process and cache the dataset
        print(f"🔄 Processing {split} dataset...")
        self._process_and_cache(dataset_name, max_length, split)
        print(f"💾 Saved processed {split} dataset to {self.cache_path}")

    # def _try_load_cached(self):
    #     """Attempt to load a cached version of the processed dataset"""
    #     if os.path.exists(self.cache_path):
    #         try:
    #             # Load the cached dataset
    #             cached_dataset = load_from_disk(self.cache_path)
                
    #             # Verify the loaded data has the expected structure
    #             if len(cached_dataset) > 0 and all(
    #                 key in cached_dataset[0] 
    #                 for key in ['input_ids', 'attention_mask', 'labels']
    #             ):
    #                 self.examples = cached_dataset
    #                 self.tokenizer = AutoTokenizer.from_pretrained(
    #                     self.tokenizer.name_or_path,  # Recover tokenizer config
    #                     use_fast=True,
    #                     padding_side="left"
    #                 )
    #                 if self.tokenizer.pad_token is None:
    #                     self.tokenizer.pad_token = self.tokenizer.eos_token
    #                 return True
                
    #             print("⚠️ Cached dataset has incorrect format, reprocessing...")
    #         except Exception as e:
    #             print(f"⚠️ Error loading cached dataset: {str(e)}, reprocessing...")

    #     return False
            

    def _process_and_cache(self, dataset_name, max_length, split, batch_size=512):
        """Process raw dataset and cache it"""
        ds = load_dataset(dataset_name, split=split)
        self.max_length = max_length

        self.examples = ds.map(
            self._process_batch,
            batched=True,
            batch_size=batch_size,
            remove_columns=ds.column_names,  # Optional: remove unused columns
        )

        self._save_processed_dataset(dataset_name=dataset_name, max_length=max_length)
        print('end init')


    def _process_batch(self, batch):
        prompts = [
            (instr.strip() + ("\n" + inp.strip() if inp.strip() else "") + "\n\n" + out.strip())
            for instr, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        
        tok = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Optional: pad to max_length if needed
            return_tensors="pt",
        )
        
        return {
            "input_ids": tok.input_ids,
            "attention_mask": tok.attention_mask,
            "labels": tok.input_ids.clone(),  # For causal LM
        }
    
    def _save_processed_dataset(self, dataset_name, max_length):
        cache_dir="processed_data"
        self.cache_path = os.path.join(cache_dir, f"{dataset_name.replace('/', '_')}_{max_length}")
        os.makedirs(cache_dir, exist_ok=True)
        self.examples.save_to_disk(self.cache_path)
        print(f"Saved processed dataset to {self.cache_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
