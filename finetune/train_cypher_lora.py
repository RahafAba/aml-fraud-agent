"""
train_cypher_lora.py
--------------------
QLoRA fine-tune of Llama-3.2-3B-Instruct for Text-to-Cypher on the AML graph schema.

Designed to run on a free Google Colab T4 GPU (Runtime -> Change runtime type -> T4 GPU).
Upload cypher_train.jsonl and cypher_train_val.jsonl to the Colab session first, then
run this script top to bottom.

The base 3B model produces invalid Cypher on complex multi-hop queries. This fine-tune
trains a small LoRA adapter (~0.75% of parameters) on ~530 schema-specific examples so the
model learns the correct patterns: variable-length ring paths, UNION for either/or queries,
and proper aggregation syntax.
"""

import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

MAX_SEQ = 1024
BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
TRAIN_FILE = "cypher_train.jsonl"
VAL_FILE = "cypher_train_val.jsonl"
OUTPUT_DIR = "cypher-finetune"


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def main():
    # 1. Load the base model in 4-bit and attach LoRA adapters (this is QLoRA).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # 2. Format each example in the Llama-3 chat template. Train and serve must
    #    use the identical format, or accuracy collapses.
    eos = tokenizer.eos_token

    def format_example(row):
        return (
            f"<|start_header_id|>system<|end_header_id|>\n\n{row['instruction']}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{row['input']}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n{row['output']}{eos}"
        )

    train_ds = Dataset.from_list(
        [{"text": format_example(r)} for r in load_jsonl(TRAIN_FILE)])
    val_ds = Dataset.from_list(
        [{"text": format_example(r)} for r in load_jsonl(VAL_FILE)])
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    # 3. Train for 3 epochs. Watch the training loss fall from ~2 toward ~0.05.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir="outputs",
            report_to="none",
        ),
    )
    trainer.train()

    # 4. Export to GGUF q4_k_m so the adapter can be loaded into Ollama on a CPU laptop.
    model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"Saved GGUF to ./{OUTPUT_DIR} -- download the .gguf and load it into Ollama.")


if __name__ == "__main__":
    main()
