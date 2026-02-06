# train_peft.py
import argparse
import json
from datasets import load_dataset, Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def make_prompt(example):
    instr = example.get("instruction","")
    inp = example.get("input","")
    out = example.get("output","")
    prompt = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return prompt

def tokenize_function(examples, tokenizer, max_length=1024):
    texts = [make_prompt(x) for x in examples]
    tokenized = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/lora")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Loading model in 8-bit (bitsandbytes)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=True,
        device_map="auto"
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    print("Loading dataset:", args.dataset_path)
    try:
        dataset = load_dataset("json", data_files=args.dataset_path)
        train_ds = dataset["train"]
    except Exception as e:
        data = []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data.append(json.loads(line))
        train_ds = Dataset.from_list(data)

    print("Tokenizing...")
    tokenized = train_ds.map(lambda ex: tokenize_function(ex, tokenizer), batched=True)
    tokenized = tokenized.remove_columns([c for c in tokenized.column_names if c not in ["input_ids","attention_mask","labels"]])
    tokenized.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
    )

    data_collator = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    print("Saving adapter...")
    model.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
