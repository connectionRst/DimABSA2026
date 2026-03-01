from datasets import load_dataset
from tqdm import tqdm

import argparse
import os

import mymod

os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"
PREFIX = mymod.prefix
MODEL_TYPES = mymod.models
DOMAIN_LANG = mymod.domain_lang

# task config
lr = 2e-4  # unsloth: use 2e-4 if the data is small
model_type = "gemma-3"

parser = argparse.ArgumentParser()
parser.add_argument('train_or_infer', default="infer")
# change what domain you want
parser.add_argument('domain', nargs='?', default="laptop")
# change the language you want to test
parser.add_argument('prd_lang', nargs='?', default="jpn")
parser.add_argument('--model-type', default="gemma-3")
# unsloth: use 2e-4 if the data is small
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--task', type=int, default=2)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('-p', type=float, default=0.8)
parser.add_argument('-k', type=int, default=20)
parser.add_argument('--temp', type=float, default=0.7)
args = parser.parse_args()

assert args.train_or_infer in ("train", "infer")
assert args.task in (1, 2, 3)  # TODO support task1

# you can change the model here
if args.train_or_infer == "train":
    if args.model_type == "gemma-3":
        model_id = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit"
    elif args.model_type == "qwen-3":
        model_id = "Qwen/Qwen3-4B-Instruct-2507"
elif args.train_or_infer == "infer":
    # TODO model id provider
    model_id = f"./Lora/{args.model_type}/{args.task}_{args.domain}/model"
else: raise ValueError("unsupport --model-type")

def train(model_id, model_type, lr, task, domain, *, resume=False):
    print(f"{model_id=}, {model_type=}, {lr=}, {task=}, {domain=}, {resume=}")

    lora_ckpt_path = './Lora/{}/{}_{}'.format(model_type, task, domain)
    train_urls = [f"{PREFIX}/subtask_{task}/{lang}/{lang}_{domain}_train_alltasks.jsonl" for lang in DOMAIN_LANG[domain]]

    # load train data from url
    dataset = load_dataset("json", data_files=train_urls)

    # Display the data info
    print(dataset)
    print(dataset['train'][0]) # type: ignore

    # ### Step 4: Load the LLM from Hugging Face and apply LoRA for fine-tuning
    # tokenizer and model setting
    model, tokenizer = mymod.get_model(model_type, model_id)

    # covert dataset to train template
    # yes, as tokenizer need to be loaded first, this cannot be raised top
    def task2prompt_mapper(x):
        text = x["Text"]
        quads = x.get("Quadruplet", [])
        if task == 2:
            answer = ", ".join([
                f"({q['Aspect']}, {q['Opinion']}, {q['VA']})"
                for q in quads
            ])
            instruction = mymod.get_instruction_task2()
        elif task == 3:
            answer = ", ".join((
                f"({q['Aspect']}, {q['Category']}, {q['Opinion']}, {q['VA']})"
                for q in quads
            ))
            instruction = mymod.get_instruction_task3(domain)
        else: raise ValueError("invalid task", task)
        prompt = instruction + "[Text] " + text + "\n\nOutput:"
        msg = mymod.wrap_prompt(model_type, prompt, answer)
        return { "text": tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=False,
                    add_special_tokens=False
                )
            }
    train_dataset = dataset["train"].map(task2prompt_mapper) # type: ignore

    # show your template text
    print(train_dataset[100]["text"]) # type: ignore

    # ### Step 5: Start LoRA Fine-Tuning
    from trl import SFTTrainer, SFTConfig # pyright: ignore[reportPrivateImportUsage]

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer, # pyright: ignore[reportCallIssue]
        train_dataset = train_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = 2, #
            learning_rate = lr, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "none", # Use TrackIO/WandB etc
            output_dir = f"./Lora/{model_type}/{task}_{domain}",
            save_strategy = "steps",
            save_steps = 100,
        ),
    )

    #print(trainer.train_dataset[100]["input_ids"])

    # TODO refractor this to mymod.py module
    from unsloth.chat_templates import train_on_responses_only
    if model_type in ("gemma-3"):
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


    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(lora_ckpt_path + "/model")  # Local saving
    tokenizer.save_pretrained(lora_ckpt_path + "/model")

def infer(model_id, model_type, task, domain, lang, top_p, top_k, temp):
    print(f"{model_id=}, {model_type=}, {task=}, {domain=}, {lang=}")
    assert lang in DOMAIN_LANG[domain]
    if task == 2:
        instruction = mymod.get_instruction_task2()
        key = "Triplet"
    elif task == 3:
        instruction = mymod.get_instruction_task3(domain)
        key = "Quadruplet"
    else: raise ValueError("not supported task")

    predict_url = f"{PREFIX}/subtask_{task}/{lang}/{lang}_{domain}_dev_task{task}.jsonl"

    # load dev json to predict
    predict_dataset = load_dataset("json", data_files=predict_url)

    # Display the data info
    print(predict_dataset["train"][0]) # type: ignore

    model, tokenizer = mymod.get_model(model_type, model_id, fast_infer=True)

    # ### Step 6: Run Inference and Extract Structured Sentiment Outputs

    # Perform inference
    results = []
    for i, sample in enumerate(tqdm(predict_dataset["train"])):
        text = sample["Text"]
        final_prompt = instruction + '[Text] ' + text + '\n\nOutput:'
        messages = mymod.wrap_prompt(model_type, final_prompt)

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
            temperature=temp, top_p=top_p, top_k=top_k,
        )

        decoded = tokenizer.decode(result[0])  #FIXME output is None
        extracted_text = decoded.split("\n")[-1]

        dump_data = {
            "ID": sample.get("ID", f"sample_{i}"),
            "Text": sample["Text"],
            key: mymod.extract_answer(extracted_text, task),
        }

        tqdm.write(repr(dump_data))
        results.append(dump_data)


    # ### Step 7: Save prediction results
    # TODO convert to def?
    import json
    import os

    # resolve output name
    out_name = f"pred_{lang}_{domain}.jsonl"

    # ensure output folder
    os.makedirs(f"subtask_{task}", exist_ok=True)

    # JSONL file path
    jsonl_path = os.path.join(f"subtask_{task}", out_name)

    # write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# main()
if args.train_or_infer == "train":
    train(model_id, args.model_type, args.lr, args.task, args.domain, resume=args.resume)
elif args.train_or_infer == "infer":
    infer(model_id, args.model_type, args.task, args.domain, args.prd_lang, args.p, args.k, args.temp)
else: raise ValueError("unsupport operation")
