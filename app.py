import os
import logging
import yaml
from fine_tuning.lora import train_lora
from fine_tuning.rag import train_rag
from fine_tuning.cag import train_cag
from utils.evaluation import evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.dataset import InstructionDataset

logger = logging.getLogger("__main__")


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_model():
    logger.info("Load models...")
    models_config = load_config("config/models.yaml")
    model_name = models_config['models'][0]['repo']  # Example: selecting the first model
    model = AutoModelForCausalLM.from_pretrained(model_name) # load_in_4bit=True for qlora
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    target_modules =models_config['models'][0]['target_modules']

    return model, tokenizer, target_modules

def load_data():
    logger.info("Load dataset...")
    datasets_config = load_config("config/datasets.yaml")
    dataset_name = datasets_config['datasets'][0]['name']  # Example: selecting the first dataset

    models_config = load_config("config/models.yaml")
    model_name = models_config['models'][0]['repo']  # Example: selecting the first model

    dataset = InstructionDataset(dataset_name, model_name, max_length=1024)

    return dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]     
    )

def main():
    setup_logging()

    logger.info("Start fine-tuning pipeline...")
    logger.info("Load fine-tuning configs...")
    fine_tuning_config = load_config("config/fine_tuning.yaml")
   
    output_dir = "./models/fine_tuned"
    os.makedirs(output_dir, exist_ok=True)

    datasets_config = load_config("config/datasets.yaml")
    dataset_name = datasets_config['datasets'][0]['name']  # Example: selecting the first dataset
    
    dataset = load_data()
    model, tokenizer, target_modules = load_model()

    # lora = train_lora(model_name, dataset_name, output_dir, fine_tuning_config['fine_tuning_methods']['lora'])
    # rag = train_rag(model_name, dataset_name, output_dir, fine_tuning_config['fine_tuning_methods']['rag'])
    # cag = train_cag(model_name, dataset_name, output_dir, fine_tuning_config['fine_tuning_methods']['cag'])

    lora = train_lora(model, tokenizer, target_modules, dataset, output_dir)    
    print(5)

    # rag = train_rag(model, tokenizer, target_modules, dataset,  output_dir)
    # cag = train_cag(model, tokenizer, target_modules, dataset,  output_dir)

    # models = {
    #     "LoRA": lora,
    #     "RAG": rag,
    #     "CAG": cag
    # }
    # best_model_name, best_model = evaluate_model(models, dataset)

    # logging.info(f"Best model: {best_model_name}")
    # best_model.save_pretrained(os.path.join(output_dir, best_model_name))

if __name__ == "__main__":
    main()
