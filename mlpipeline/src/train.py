from cProfile import run
from config import cmd_arguments

from prometheus_client import MetricsHandler
import numpy as np
from wandb.keras import WandbCallback

# from transformers.keras_callbacks import KerasMetricCallback
from transformers import DataCollatorForSeq2Seq, TFAutoModelForSeq2SeqLM, AdamWeightDecay
from transformers.keras_callbacks import PushToHubCallback

import cocitedata
import train_helpers
import keras_metric_callback
from train_helpers import CustomMetrics, SaveModelCallback, tokens_2_words

import wandb
from codecarbon import EmissionsTracker
import re


def main():
    args = cmd_arguments()

    wandb.init(project="cocite", config=args, mode=args.wandb_mode)
    # TODO make from_pytorch dynamic. if tensorflow model or hub model set to false. else True.
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.modelname, from_pt=args.from_pytorch)
    tokenized_datasets, tokenizer = cocitedata.load_dataset(args, model_name_or_path=model.name_or_path)

    # initialize model

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
    tokenized_train = tokenized_datasets["train"]
    tokenized_test = tokenized_datasets["test"]
    if args.debug:
        tokenized_train = tokenized_train.select(list(range(2)))
        tokenized_test = tokenized_test.select(list(range(2)))

    tf_train_set = tokenized_train.to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=True,
        batch_size=args.batchsize,
        collate_fn=data_collator,
    )

    tf_test_set = tokenized_test.to_tf_dataset(
        columns=["attention_mask", "input_ids", "labels"],
        shuffle=True,
        batch_size=args.batchsize,
        collate_fn=data_collator,
        
    )

    # valid number between 100 and 5000
    ds_len = len(tokenized_test["label"])
    num_demo_samples = max(100, ds_len // 10)
    num_demo_samples = min(10000, num_demo_samples)
    if num_demo_samples > ds_len:
        num_demo_samples = ds_len

    generation_test_dataset = (
        tokenized_test
        .shuffle()
        .select(list(range(num_demo_samples)))
        .to_tf_dataset(
            batch_size=args.batchsize,
            drop_remainder=True,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )
    generation_train_dataset = (
        tokenized_train
        .shuffle()
        .select(list(range(num_demo_samples)))
        .to_tf_dataset(
            batch_size=args.batchsize,
            drop_remainder=True,
            columns=["input_ids", "attention_mask", "labels"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )

    top_ks = [1, 3, 5, 20]
    max_k = max(top_ks)
    max_k = min(max_k, args.topk) 
    top_ks = [k for k in top_ks if k <= max_k]


    metric_fn_test = CustomMetrics(prefix="test_", args=args, top_ks=top_ks).fast_metrics
    metric_fn_train = CustomMetrics(prefix="train_", args=args, top_ks=top_ks).fast_metrics
    metric_test_callback = keras_metric_callback.KerasMetricCallback(
        model=model,
        tokenizer=tokenizer,
        metric_fn=metric_fn_test,
        eval_dataset=generation_test_dataset, prefix="test_",
        predict_with_generate=True, args=args, batch_size=args.batchsize,
        len_train_dataset = len(tokenized_train["label"]),
        top_ks=top_ks,
    )
    metric_train_callback = keras_metric_callback.KerasMetricCallback(
        model=model,
        tokenizer=tokenizer,
        metric_fn=metric_fn_train,
        eval_dataset=generation_train_dataset, prefix="train_",
        predict_with_generate=True, args=args, batch_size=args.batchsize,
        len_train_dataset = len(tokenized_train["label"]),
        top_ks=top_ks,
    )
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)  # TODO warning high lr
    model.compile(optimizer=optimizer)

    wandb_callback = WandbCallback(save_model=not args.debug)

    callbacks = [
        wandb_callback,
        metric_test_callback, 
        metric_train_callback,
    ]


    if True or not args.debug:
        modelsave_dir="../model_save/" + args.modelname + "_" + str(wandb.run.id) + "/"
        modelsave_dir += "debug/" if args.debug else ""
        training_step = 0
        if "model_save" in args.modelname:
            # load local checkpoint instead of huggingface hub model
            # set number of train steps if possible
            if "_steps_" in args.modelname:
                # find first int in string after substring "_steps_"
                steps_text = args.modelname.split("_steps_")[-1]
                training_step = int(re.search(r'\d+', steps_text).group())

        save_model_callback = SaveModelCallback(
            modelsave_dir, model=model,
            tokenizer=tokenizer, training_step=training_step)
        callbacks.append(save_model_callback)


    if not args.notraining:
        if not args.debug:
            tracker = EmissionsTracker()
            tracker.start()
        model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=args.epochs, callbacks=callbacks)
        # GPU Intensive code goes here
        if not args.debug:
            co2_emissions = tracker.stop()
            wandb.log({"CO2_emissions (in Kg)": co2_emissions})

    ### evaluate model
    if not args.noevaluation:
        train_helpers.evaluate(model, generation_test_dataset, metric_fn_test, prefix="test_", args=args, top_ks=top_ks,
                            tokenizer=tokenizer)
        train_helpers.evaluate(model, generation_train_dataset, metric_fn_train, prefix="train_", args=args, top_ks=top_ks,
                            tokenizer=tokenizer)

if __name__ == "__main__":
    main()