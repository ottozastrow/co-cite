from cProfile import run
from config import cmd_arguments
args = cmd_arguments()

from prometheus_client import MetricsHandler
import numpy as np
from wandb.keras import WandbCallback

# from transformers.keras_callbacks import KerasMetricCallback
from transformers import DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, AdamWeightDecay
from transformers.keras_callbacks import PushToHubCallback

import cocitedata
import train_helpers
import keras_metric_callback
from train_helpers import CustomMetrics, SaveModelCallback

import wandb

wandb.init(project="cocite", config=args, mode=args.wandb_mode)

tokenized_datasets, tokenizer = cocitedata.load_dataset(args)

# initialize model
model = TFAutoModelForSeq2SeqLM.from_pretrained(args.modelname)

# tokenized_datasets = tokenized_datasets.remove_columns(dataset["train"].column_names)
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

# valid number between 100 and 1000
ds_len = len(tokenized_datasets["test"]["label"])
num_demo_samples = max(100, ds_len // 10)
num_demo_samples = min(5000, num_demo_samples)
if num_demo_samples > ds_len:
    num_demo_samples = ds_len


generation_test_dataset = (
    tokenized_datasets["test"]
    .shuffle()
    .select(list(range(num_demo_samples)))
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
    .select(list(range(num_demo_samples)))
    .to_tf_dataset(
        batch_size=args.batchsize,
        columns=["input_ids", "attention_mask", "labels"],
        shuffle=False,
        collate_fn=data_collator,
    )
)


metric_fn_test = CustomMetrics(prefix="test_", args=args).fast_metrics
metric_fn_train = CustomMetrics(prefix="train_", args=args).fast_metrics
metric_test_callback = keras_metric_callback.KerasMetricCallback(
    tokenizer=tokenizer,
    metric_fn=metric_fn_test,
    eval_dataset=generation_test_dataset, prefix="test_",
    predict_with_generate=True, args=args, batch_size=args.batchsize,
)
metric_train_callback = keras_metric_callback.KerasMetricCallback(
    tokenizer=tokenizer,
    metric_fn=metric_fn_train,
    eval_dataset=generation_train_dataset, prefix="train_",
    predict_with_generate=True, args=args, batch_size=args.batchsize,
)
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)  # TODO warning high lr
model.compile(optimizer=optimizer)

wandb_callback = WandbCallback(save_model=not args.debug)


callbacks = [
    wandb_callback,
    metric_test_callback, 
    metric_train_callback,
]

if True: # not args.debug:
    modelsave_dir="./model_save/" + args.modelname + "_" + str(wandb.run.id) + "/"
    modelsave_dir += "debug/" if args.debug else ""
    save_model_callback = SaveModelCallback(modelsave_dir, model=model, tokenizer=tokenizer)
    callbacks.append(save_model_callback)


if not args.notraining:
    model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=args.epochs,callbacks=callbacks)

### evaluate model
if not args.noevaluation:
    print("starting eval")
    for batch in generation_test_dataset:
        inputs = batch["input_ids"]
        labels = batch["labels"]
        predictions = model.generate(inputs, num_beams=args.topk, num_return_sequences=args.topk,
                                    output_scores=True, return_dict_in_generate=True)
        results = metric_fn_test((predictions, labels))
        print({'eval_test': results})
        wandb.log({'eval_test': results})

    # for batch in generation_train_dataset:
    #     inputs = batch["input_ids"]
    #     labels = batch["labels"]
    #     predictions = model.generate(inputs, num_beams=args.topk, num_return_sequences=args.topk,
    #                                 output_scores=True, return_dict_in_generate=True)
    #     results = metric_fn_test((predictions, labels))
    #     print({'eval_train': results})
    #     wandb.log({'eval_train': results})
