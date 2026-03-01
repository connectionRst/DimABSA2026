# ### Step 1: Install unsloth package
# ### Step 2: Load the competition data
# ### Step 3: Design prompt template

from datasets import load_dataset

import os
import sys

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
PREFIX = "../../../task-dataset/track_a"
MODEL_TYPES = ("gemma3", "qwen3")

# task config
try: domain = sys.argv[1]
except: domain = "laptop" # change what domain you want to test

lr = 2e-4  # unsloth: use 2e-4 if the data is small
subtask = "subtask_2" # subtask_2 or subtask_3
task = "task2" # task2 or task3
domain_lang = {
    "laptop": ["eng", "zho"],
    "restaurant": ["eng", "rus", "tat", "ukr", "zho"],
    "hotel": ["jpn"]
}
resume_from_ckpt = True
model_type = "gemma3"

print(f"{domain=}, {model_type=}, {lr=}, {resume_from_ckpt=}")

train_urls = [f"{PREFIX}/{subtask}/{lang}/{lang}_{domain}_train_alltasks.jsonl" for lang in domain_lang[domain]]

# load train data from url
dataset = load_dataset("json", data_files=train_urls)

# Display the data info
print(dataset)
print(dataset['train'][0]) # type: ignore

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

    def convert(x, *, tokenizer):
        text = x["Text"]
        quads = x.get("Quadruplet", [])
        answer = ", ".join([
            f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
            for q in quads
        ])
        prompt = instruction + "[Text] " + text + "\n\nOutput:"
        msg = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        try:
            return {
                "text":
                    tokenizer.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    ).removeprefix('<bos>')
            }
        except:
            return msg

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

    def convert(x, *, tokenizer):
        text = x["Text"]
        quads = x.get("Quadruplet", [])
        answer = ", ".join((
            f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
            for q in quads
        ))
        prompt = instruction + "[Text] " + text + "\n\nOutput:"
        msg = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
        try:
            return {
                "text": tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False
                ).removeprefix('<bos>')
            }
        except:
            return msg


# ### Step 4: Load the LLM from Hugging Face and apply LoRA for fine-tuning
if model_type in ("gemma3"):
    from unsloth import FastModel as FastLanguageModel
else:
    from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import train_on_responses_only


# you can change the model here
model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
# model_id = "/mnt/sata-4t/modelscope-cache/models/Qwen/Qwen3-4B-Instruct-2507"

# tokenizer and model setting
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 512, # DimASBA Task usually less then 512 tokens.
    # load_in_16bit = True,
    # device_map = "balanced",
    local_files_only = True
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",  # FIXME
)

# lora setting
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    finetune_vision_layers = False, # Turn off for just text!
)

# covert dataset to train template
# FIXME yes, as tokenizer need to be loaded first, this cannot be raised to top
train_dataset = dataset["train"].map(lambda x: convert(x, tokenizer=tokenizer)) # type: ignore

# show your template text
print(train_dataset[100]["text"]) # type: ignore


# %% [markdown]
# ### Step 5: Start LoRA Fine-Tuning
# 

# %%
from trl import SFTTrainer, SFTConfig # pyright: ignore[reportPrivateImportUsage]

'''
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 20,
        num_train_epochs = 2, # epoches
        learning_rate = 1e-4, #　learning rate
        logging_steps = 50,
        save_steps = 100,
        fp16 = True,
        bf16 = False,
        report_to = "none",
        output_dir = "./Lora", # save Lora
    ),
)
'''

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer, # pyright: ignore[reportCallIssue]
    train_dataset = train_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 2, #
        learning_rate = lr, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
        output_dir = f"./Lora/gemma/{task}_{domain}",
        save_strategy = "steps",
        save_steps = 100,
    ),
)

if model_type in ("gemma3"):
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
else:
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )

#print(trainer.train_dataset[100]["input_ids"])

trainer.train(resume_from_checkpoint=resume_from_ckpt)

model.save_pretrained(f"./Lora/gemma/{task}_{domain}/model")  # Local saving
tokenizer.save_pretrained(f"./Lora/gemma/{task}_{domain}/model")
