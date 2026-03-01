# ### Step 2: Load the competition data
# ### Step 3: Design prompt template

from tqdm import tqdm
from datasets import load_dataset

import os
import sys
import re
import json

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
PREFIX = "../../../task-dataset/track_a"
MODEL_TYPES = ("qwen3", "gemma3")

# task config
try: domain = sys.argv[1]
except: domain = "hotel" #change what domain you want to test
try: prd_lang = sys.argv[2]
except: prd_lang = "jpn" #chang the language you want to test

subtask = "subtask_2" # subtask_2 or subtask_3
task = "task2" # task2 or task3
domain_lang = {
   "laptop": ["eng", "zho"],
   "restaurant": ["eng", "rus", "tat", "ukr", "zho"],
   "hotel": ["jpn"]
}
model_type = "gemma3"

print(f"{domain=}, {prd_lang=}")
assert prd_lang in domain_lang[domain]

predict_url = f"{PREFIX}/{subtask}/{prd_lang}/{prd_lang}_{domain}_dev_{task}.jsonl"

# load dev json to predict
predict_dataset = load_dataset("json", data_files=predict_url)

# Display the data info
print(predict_dataset["train"][0]) # type: ignore

# task 2 prompt template covert
if task == "task2":
    instruction = '''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

    ### Instruction:
    Given a textual instance [Text], extract all (A, O, VA) triplets, where:
    - A is an Aspect term (a phrase describing an entity mentioned in [Text])
    - O is an Opinion term
    - VA is a Valence–Arousal score in the format (valence#arousal)

    Valence ranges from 1 (negative) to 9 (positive),
    Arousal ranges from 1 (calm) to 9 (excited).

    ### Example:
    Input:
    [Text] average to good thai food, but terrible delivery.

    Output:
    [Triplet] (thai food, average to good, 6.75#6.38), (delivery, terrible, 2.88#6.62)

    ### Question:
    Now complete the following example:
    Input:
    '''

# task 3 prompt template covert, with task3 predefine entity and attribute labels.
elif task == "task3":
    rest_entity = 'RESTAURANT, FOOD, DRINKS, AMBIENCE, SERVICE, LOCATION'
    rest_attribute = 'GENERAL, PRICES, QUALITY, STYLE_OPTIONS, MISCELLANEOUS'

    laptop_entity = 'LAPTOP, DISPLAY, KEYBOARD, MOUSE, MOTHERBOARD, CPU, FANS_COOLING, PORTS, MEMORY, POWER_SUPPLY, OPTICAL_DRIVES, BATTERY, GRAPHICS, HARD_DISK, MULTIMEDIA_DEVICES, HARDWARE, SOFTWARE, OS, WARRANTY, SHIPPING, SUPPORT, COMPANY'
    laptop_attribute = 'GENERAL, PRICE, QUALITY, DESIGN_FEATURES, OPERATION_PERFORMANCE, USABILITY, PORTABILITY, CONNECTIVITY, MISCELLANEOUS'

    hotel_entity = 'HOTEL, ROOMS, FACILITIES, ROOM_AMENITIES, SERVICE, LOCATION, FOOD_DRINKS'
    hotel_attribute = 'GENERAL, PRICE, COMFORT, CLEANLINESS, QUALITY, DESIGN_FEATURES, STYLE_OPTIONS, MISCELLANEOUS'

    finance_entity = 'MARKET, COMPANY, BUSINESS, PRODUCT'
    finance_attribute = 'GENERAL, SALES, PROFIT, AMOUNT, PRICE, COST'

    entity_attribute_map = {
        'restaurant': (rest_entity, rest_attribute),
        'laptop': (laptop_entity, laptop_attribute),
        'hotel': (hotel_entity, hotel_attribute),
        'finance': (finance_entity, finance_attribute),
    }

    entity_label, attribute_label = entity_attribute_map[domain]

    instruction = f'''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

    ### Instruction:
    Given a textual instance [Text], extract all (A, C, O, VA) quadruplets, where:
    - A is an Aspect term (a phrase describing an entity mentioned in [Text])
    - C is a Category label (e.g. FOOD#QUALITY)
    - O is an Opinion term
    - VA is a Valence–Arousal score in the format (valence#arousal)

    Valence ranges from 1 (negative) to 9 (positive),
    Arousal ranges from 1 (calm) to 9 (excited).

    ### Label constraints:
    [Entity Labels] ({entity_label})
    [Attribute Labels] ({attribute_label})

    ### Example:
    Input:
    [Text] average to good thai food, but terrible delivery.

    Output:
    [Quadruplet] (thai food, FOOD#QUALITY, average to good, 6.75#6.38),
                (delivery, SERVICE#GENERAL, terrible, 2.88#6.62)

    ### Question:
    Now complete the following example:
    Input:
    '''


# show your template text
pass

# extract answer
def extract_answer(text,task):
  result = []
  if task == "task2":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
    matches = re.findall(pattern, text)

    for aspect, opinion, va in matches:
        meta_triplet = {}
        meta_triplet["Aspect"] = aspect.strip()
        meta_triplet["Opinion"] = opinion.strip()
        meta_triplet["VA"] = va
        result.append(meta_triplet)

  elif task == "task3":
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, text)

    for aspect, category, opinion, va in matches:
        meta_quadra = {}
        meta_quadra["Aspect"] = aspect.strip()
        meta_quadra["Category"] = category.strip()
        meta_quadra["Opinion"] = opinion.strip()
        meta_quadra["VA"] = va
        result.append(meta_quadra)
  else:
    raise ValueError("Invalid task")

  return result

# convert text to prompt
# TODO merge with convert(x, *, tokenizer)
def format_dataset(x, *, model_type="qwen3"):
    text = x["Text"]
    final_prompt = instruction + '[Text] ' + text + '\n\nOutput:'
    if model_type == "qwen3":
        return [
            {"role": "user", "content": final_prompt}
        ]
    elif model_type == "gemma3":
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": final_prompt
            }]
        }]
    else:
        raise ValueError("model_type is invalid")


# ### Step 4: Load the LLM from Hugging Face and apply LoRA for fine-tuning
if model_type in ("gemma3"):
    from unsloth import FastModel as FastLanguageModel
else:
    from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

model_id = f"./Lora/gemma/{task}_{domain}/model"

# tokenizer and model setting
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 512, # DimASBA Task usually less then 512 tokens.
    # load_in_16bit = True,  # gemma not support 16bit
    # device_map = "balanced",
    local_files_only = True
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

FastLanguageModel.for_inference(model)

# ### Step 6: Run Inference and Extract Structured Sentiment Outputs

# Perform inference
results = []
for i, sample in enumerate(tqdm(predict_dataset["train"])):
    messages = format_dataset(sample, model_type="gemma3")

    text = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
        return_dict = True
    )

    result = model.generate(
        **text.to("cuda"),
        max_new_tokens=1024,
        temperature=0.7, top_p=0.8, top_k=20,
    )

    decoded = tokenizer.decode(result[0])  #FIXME output is None
    extracted_text = decoded.split("\n")[-1]

    key = "Triplet" if task == "task2" else "Quadruplet"

    dump_data = {
        "ID": sample.get("ID", f"sample_{i}"),
        "Text": sample["Text"],
        key: extract_answer(extracted_text, task),
    }

    tqdm.write(repr(dump_data))
    results.append(dump_data)


# ### Step 7: Save prediction results
# TODO convert to def?
import json
import os

# resolve output name
out_name = f"pred_{prd_lang}_{domain}.jsonl"

# ensure output folder
os.makedirs(subtask, exist_ok=True)

# JSONL file path
jsonl_path = os.path.join(subtask, out_name)

# write JSONL
with open(jsonl_path, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

