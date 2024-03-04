from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from repe import repe_pipeline_registry, WrappedReadingVecModel

repe_pipeline_registry()


def convert_array(arr):
    return [1 if x > 128 else 0 for x in arr]


model_name_or_path = "/data/dataset/llama/Llama-2-13b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.float16, device_map="balanced_low_0"
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
tokenizer.padding_side = "left"
tokenizer.pad_token = (
    tokenizer.unk_token if tokenizer.pad_token is None else tokenizer.pad_token
)


template = "[INST] <<SYS>>\nYou are a helpful, highly skilled assistant. Always answer as short as possible, while being accurate. Your answers should not not exceed 128 tokens.\n<</SYS>>\n\n{instruction} [/INST] "

# dataset = load_dataset("justinphan3110/harmful_harmless_instructions")

import json

# 要提取的键列表
keys_to_extract = ["question", "output_token_length"]

# 初始化一个字典，以存储每个键对应的值列表
pos_values = {key: [] for key in keys_to_extract}
neg_values = {key: [] for key in keys_to_extract}

# 读取JSON文件
with open(
    "/data/siqizhu/representation-engineering/data/eos/llama_outputs.json", "r"
) as file:
    data = json.load(file)

import random


def shuffle_list(original_list):
    # Copy the original list to avoid modifying it directly
    shuffled_list = original_list[:]
    # Use random.shuffle to shuffle the list in place
    random.shuffle(shuffled_list)
    return shuffled_list


# Shuffle the example list
shuffled_data = shuffle_list(data)

# 遍历列表中的每个字典
for item in shuffled_data:
    if item["output_token_length"] > 128:
        for key in keys_to_extract:
            # 提取并存储相应的值
            neg_values[key].append(item[key])
    else:
        for key in keys_to_extract:
            # 提取并存储相应的值
            pos_values[key].append(item[key])


# 打印结果，或根据需要进一步处理
# print(extracted_values)

length = min(len(pos_values["question"]), len(neg_values["question"]))
train_length = int(0.8 * length)
test_length = length - train_length

pos_input = pos_values["question"][:length]
neg_input = neg_values["question"][:length]
pos_label = [True for _ in range(length)]
neg_label = [False for _ in range(length)]

assert len(pos_input) == len(pos_label)
print(len(pos_input))
# Combine lists into a 2D list
input = [[pos_input[i], neg_input[i]] for i in range(len(pos_input))]

input_label = [[True, False] for i in range(len(pos_input))]

train_data, train_labels = input[:train_length], input_label[:train_length]
# Convert the example array
# real_train_labels = convert_array(train_labels)
print(len(train_data), len(train_labels))

test_data = input[train_length:]

print(len(test_data))

train_data = np.concatenate(train_data).tolist()
test_data = np.concatenate(test_data).tolist()

train_data = [template.format(instruction=s) for s in train_data]
test_data = [template.format(instruction=s) for s in test_data]

print(train_data[0])

rep_token = -1
hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
n_difference = 1
direction_method = "pca"
rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)

direction_finder_kwargs = {"n_components": 1}

rep_reader = rep_reading_pipeline.get_directions(
    train_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    n_difference=n_difference,
    train_labels=train_labels,
    direction_method=direction_method,
    direction_finder_kwargs=direction_finder_kwargs,
)

component_index = 0

H_tests = rep_reading_pipeline(
    test_data,
    rep_token=rep_token,
    hidden_layers=hidden_layers,
    rep_reader=rep_reader,
    component_index=component_index,
    batch_size=32,
)

results = {layer: {} for layer in hidden_layers}
for layer in hidden_layers:
    H_test = [H[layer] for H in H_tests]
    H_test = [H_test[i : i + 2] for i in range(0, len(H_test), 2)]

    sign = rep_reader.direction_signs[layer][component_index]
    eval_func = min if sign == -1 else max

    cors = np.mean([eval_func(H) == H[0] for H in H_test])
    results[layer] = cors

x = list(results.keys())
y = [results[layer] for layer in results]
plt.plot(x, y)

plt.savefig("/data/siqizhu/representation-engineering/examples/eos/eos_plot1.png")
