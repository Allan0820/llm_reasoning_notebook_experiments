import pandas as pd 
import torch
import os 
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset 
from huggingface_hub import login
TOKEN = os.getenv("TOKEN")
login(token = TOKEN)

def run_model(models, DATASET_PATH, TEST_SIZE, SEED):
   
    for model in models:
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model)

        dataset = load_dataset('csv', data_files = DATASET_PATH )['train']
        split_dataset = dataset.train_test_split(test_size = TEST_SIZE, seed = SEED)

        train_set = split_dataset['train']
        test_set = split_dataset['test']

        train_tokenized = train_set.map(preprocess, batched=True)
        test_tokenized = test_set.map(preprocess, batched=True)

        training_args = TrainingArguments(
            
                output_dir = "./results",
                per_device_train_batch_size = 12,
                per_device_eval_batch_size = 12,
                eval_strategy = 'epoch',
                save_strategy = 'epoch',
                do_eval = True,
                logging_steps = 2,
                eval_steps = 2, 
                num_train_epochs = 1,
                learning_rate = 2e-5,

        )

        trainer = Trainer(
            
                model = model,
                args = training_args,
                train_dataset = train_tokenized,
                eval_dataset = test_tokenized,

        )

        trainer.train()

        # Post training 

        # model.save_pretrained("/usr3/allanp/llm_reasoning_notebook_experiments/model_ckpts", from_pt=True) 

        # Model evaluation 

        # Clear data usage 
        del model, tokenizer
        torch.cuda.synchronize()  # Soft stop the GPU and ensure all processes finish
        torch.cuda.empty_cache()  # Clear GPU RAM before starting the next execution


#====================================
# Function implementations 

def preprocess(batch):
     inputs = tokenizer(
         batch['natural_language'],
         max_length=128,
         truncation=True,
         padding = 'max_length',
         
     )    
    
     labels = tokenizer(
         batch['sympy'],
         max_length =128,
         truncation = True, 
         padding = 'max_length'
     )
     inputs['labels']=labels['input_ids']
     return inputs
 
 
 #unit test for the function
 
run_model("google/gemma-3-1b-it", 'fol_sympy_nl_16k.csv', 0.2, 42)