from prometheus_client import MetricsHandler
import wandb
import numpy as np
from wandb.keras import WandbCallback

from datasets import load_metric
from transformers.keras_callbacks import KerasMetricCallback
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
# model = AutoModelForSeq2SeqLM.from_pretrained(args.modelname)
from transformers import TFAutoModelForSeq2SeqLM, AdamWeightDecay
model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-small")
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
generation_dataset = (
    tokenized_datasets["test"]
    .shuffle()
    .to_tf_dataset(
        batch_size=args.batchsize,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)

rouge_metric = load_metric("rouge")

def rouge_fn(data):
    inputs, labels = data
    predictions = model.generate(inputs)
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    ### Compute ROUGE
    result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
    results =  {key: value.mid.fmeasure * 100 for key, value in result.items()}
    wandb.log({"rouge": results})

    ### accuracy
    results['acc'] = np.mean([pred == label for pred, label in list(zip(decoded_predictions, decoded_labels))])
    wandb.log({"accuracy": results['acc']})

    ### sample outputs
    my_table = wandb.Table(columns=["inputs", "prediction", "groundtruth"], 
    data=[list(t) for t in zip(decoded_inputs, decoded_predictions, decoded_labels)])
    wandb.log({"demo": my_table})
    return results

# def metric_fn(eval_predictions):
#     preds, labels = eval_predictions
#     prediction_lens = [
#         np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
#     ]
#     result["gen_len"] = np.mean(prediction_lens)
#     return result


metric_callback = KerasMetricCallback(
    metric_fn=rouge_fn, eval_dataset=generation_dataset, predict_with_generate=True
)
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
if not args.notraining:
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=args.epochs,
              callbacks=[WandbCallback(), metric_callback])

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

