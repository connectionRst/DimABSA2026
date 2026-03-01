prefix = "../../../task-dataset/track_a"
models = ("gemma-3", "qwen-3")
moe_models = ("gemma-3")
domain_lang = {
    "laptop": ["eng", "zho"],
    "restaurant": ["eng", "rus", "tat", "ukr", "zho"],
    "hotel": ["jpn"]
}

# TODO save model_type in a function.
model_type = None
def get_model(model_type, model_path, *, fast_infer=False):
    '''
    return (model w/ lora, tokenizer)
    '''
    if model_type in moe_models:
        from unsloth import FastModel as FastLanguageModel
    else:
        from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    # tokenizer and model setting
    # TODO? for qwen3 load in 16bit
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 512, # DimASBA Task usually less then 512 tokens.
        # load_in_16bit = True,
        # device_map = "balanced",
        local_files_only = True
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = model_type,
    )

    if fast_infer: FastLanguageModel.for_inference(model)
    else:
        if model_type == "gemma-3":
            # lora setting
            model = FastLanguageModel.get_peft_model(
                model,
                r = 8,
                lora_alpha = 8,
                lora_dropout = 0,
                finetune_vision_layers = False, # Turn off for just text!
            )
        elif model_type == "qwen-3":
            model = FastLanguageModel.get_peft_model(
                model,
                r = 8,
                lora_alpha = 8,
                lora_dropout = 0,
            )
        else: raise ValueError("invalid model_type", model_type)
    return model, tokenizer

def get_instruction_task2():
    instruction = \
    '''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
    return instruction

def get_instruction_task3(domain):
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

    instruction = \
    f'''Below is an instruction describing a task, paired with an input that provides additional context. Your goal is to generate an output that correctly completes the task.

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
    return instruction

# extract answer
def extract_answer(text, task):
    import re
    result = []
    if task == 2:
        pattern = r'\(([^,]+),\s*([^,]+),\s*([\d.]+#[\d.]+)\)'
        matches = re.findall(pattern, text)

        for aspect, opinion, va in matches:
            meta_triplet = {}
            meta_triplet["Aspect"] = aspect.strip()
            meta_triplet["Opinion"] = opinion.strip()
            meta_triplet["VA"] = va
            result.append(meta_triplet)

    elif task == 3:
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

def is_multimodal(model_type):
    NATIVE_MULTIMODAL_LIST = ("gemma-3")
    return model_type in NATIVE_MULTIMODAL_LIST

def wrap_prompt(model_type, prompt, answer=None):
    prompt = [{
            "role": "user",
            "content": prompt
        }]
    if answer is not None:
        prompt.append({"role": "assistant", "content": answer})

    # FIXME is it needed to split this logic to separate function for more strict
    # typeguard, or as user only need the template so type is not necessary?
    if is_multimodal(model_type):
        for p in prompt:
            p["content"] = [{"type": "text", "text": p["content"]}]
    return prompt

def to_task_prompt_mapper(x, *, task, tokenizer):
    text = x["Text"]
    quads = x.get("Quadruplet", [])
    if task == 2:
        answer = ", ".join([
            f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
            for q in quads
        ])
        instruction = get_instruction_task2()
    elif task == 3:
        answer = ", ".join((
            f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
            for q in quads
        ))
        instruction = get_instruction_task3()
    else: raise ValueError("invalid task", task)
    prompt = instruction + "[Text] " + text + "\n\nOutput:"
    msg = wrap_prompt(prompt, answer)
    try:
        return {
            "text":
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=False,
                add_special_tokens=False
            )
        }
    except:
        return msg
