import wandb
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer

import cocitedata
import train_helpers
from config import cmd_arguments

args = cmd_arguments()
wandb.init(project="cocite")
wandb.config.update(args)


dataset = cocitedata.load_dataset(args)

# initialize model
model = AutoModelForSeq2SeqLM.from_pretrained(args.modelname)
tokenizer = AutoTokenizer.from_pretrained(args.modelname)

tokenized_datasets = dataset.map(
    train_helpers.create_tokenize_function(tokenizer), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
# tokenized_datasets = tokenized_datasets.remove_columns(dataset["test"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    eval_accumulation_steps=8,
    per_device_train_batch_size=args.batchsize,
    per_device_eval_batch_size=args.batchsize,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.epochs,
    fp16=False,
    report_to="wandb"
)
import pdb

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=train_helpers.create_metrics(tokenizer),
)

for i in range(10):
    if not args.notraining:
        trainer.train()

    predictions = trainer.predict(tokenized_datasets["test"])
    results = train_helpers.create_metrics(tokenizer)(predictions)
    wandb.log({'eval_test': results})

    predictions = trainer.predict(tokenized_datasets["train"])
    results = train_helpers.create_metrics(tokenizer)(predictions)
    wandb.log({'eval_train': results})

