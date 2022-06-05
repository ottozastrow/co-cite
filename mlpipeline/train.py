from config import cmd_arguments
args = cmd_arguments()

from prometheus_client import MetricsHandler
import numpy as np
from wandb.keras import WandbCallback

# from transformers.keras_callbacks import KerasMetricCallback
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, AdamWeightDecay

import cocitedata
import train_helpers
import keras_metric_callback

import wandb
wandb.init(project="cocite")
wandb.config.update(args)


dataset = cocitedata.load_dataset(args)

# initialize model
model = TFAutoModelForSeq2SeqLM.from_pretrained(args.modelname)
tokenizer = AutoTokenizer.from_pretrained(args.modelname)

tokenized_datasets = dataset.map(
    train_helpers.create_tokenize_function(tokenizer), batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
# tokenized_datasets = tokenized_datasets.remove_columns(dataset["test"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")

tf_train_set = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=args.batchsize,
    collate_fn=data_collator,
)

tf_test_set = tokenized_datasets["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=False,
    batch_size=args.batchsize,
    collate_fn=data_collator,
    
)
generation_test_dataset = (
    tokenized_datasets["test"]
    .shuffle()
    .select(list(range(args.miniature_dataset_size//10+1)))
    .to_tf_dataset(
        batch_size=args.batchsize,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)
generation_train_dataset = (
    tokenized_datasets["train"]
    .shuffle()
    .select(list(range(args.miniature_dataset_size//10+1)))
    .to_tf_dataset(
        batch_size=args.batchsize,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)


metric_test_callback = keras_metric_callback.KerasMetricCallback(
    metric_fn=train_helpers.create_metrics_fn(prefix="test_", tokenizer=tokenizer, model=model, args=args), 
        eval_dataset=generation_test_dataset, predict_with_generate=True, num_beams=3
)
metric_train_callback = keras_metric_callback.KerasMetricCallback(
    metric_fn=train_helpers.create_metrics_fn(prefix="train_", tokenizer=tokenizer, model=model, args=args), 
        eval_dataset=generation_train_dataset, predict_with_generate=True, num_beams=3
)
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
if not args.notraining:
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=args.epochs,
              callbacks=[WandbCallback(), metric_test_callback, metric_train_callback])

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="steps",
#     learning_rate=2e-5,
#     eval_accumulation_steps=8,
#     per_device_train_batch_size=args.batchsize,
#     per_device_eval_batch_size=args.batchsize,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=args.epochs,
#     fp16=False,
#     report_to="wandb"
# )
# import pdb

# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=train_helpers.create_metrics(tokenizer),
# )

# if not args.notraining:
#     trainer.train()

# predictions = trainer.predict(tokenized_datasets["test"])
# results = train_helpers.create_metrics(tokenizer)(predictions)
# wandb.log({'eval_test': results})

# predictions = trainer.predict(tokenized_datasets["train"])
# results = train_helpers.create_metrics(tokenizer)(predictions)
# wandb.log({'eval_train': results})

