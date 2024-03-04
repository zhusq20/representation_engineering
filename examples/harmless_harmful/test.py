from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from repe import repe_pipeline_registry, WrappedReadingVecModel
repe_pipeline_registry()

model_name_or_path = '/data/dataset/llama/Llama-2-13b-chat-hf'

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    device_map='balanced_low_0'
    ).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token

template =  "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{instruction} [/INST] "
dataset = load_dataset("justinphan3110/harmful_harmless_instructions")

train_dataset, test_dataset = dataset['train'], dataset['test'] if 'test' in dataset else dataset['train']

train_data, train_labels = train_dataset['sentence'], train_dataset['label']
test_data = test_dataset['sentence']

train_data = np.concatenate(train_data).tolist()
test_data = np.concatenate(test_data).tolist()

train_data = [template.format(instruction=s) for s in train_data]
test_data = [template.format(instruction=s) for s in test_data]

print(train_data[0])
print(train_labels[0])

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = 'pca'
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

direction_finder_kwargs={"n_components": 1}

rep_reader = rep_reading_pipeline.get_directions(
    train_data, 
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    n_difference=n_difference, 
    train_labels=train_labels, 
    direction_method=direction_method,
    direction_finder_kwargs=direction_finder_kwargs
)

component_index=0

H_tests = rep_reading_pipeline(
    test_data,
    rep_token=rep_token, 
    hidden_layers=hidden_layers, 
    rep_reader=rep_reader,
    component_index=component_index,
    batch_size=32)

results = {layer: {} for layer in hidden_layers}
for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests] 
    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
    
    sign = rep_reader.direction_signs[layer][component_index]
    eval_func = min if sign == -1 else max
    
    cors = np.mean([eval_func(H) == H[0] for H in H_test])
    results[layer] = cors

x = list(results.keys())
y = [results[layer] for layer in results]
plt.plot(x, y)

