
from codecarbon import EmissionsTracker
from transformers import DataCollatorForSeq2Seq, \
    TFAutoModelForSeq2SeqLM, AdamWeightDecay
import wandb
from wandb.keras import WandbCallback

import cocitedata
import train_helpers
import callbacks
from config import cmd_arguments
from callbacks import SaveModelCallback


def main():
    args = cmd_arguments()
    if args.tags:
        tags = args.tags.split(",")
    else:
        tags = None
    wandb.init(project="cocite", config=args, mode=args.wandb_mode, tags=tags, name=args.runname)

    # TODO make from_pytorch dynamic. if tensorflow model or hub model set to false. else True.
    model = TFAutoModelForSeq2SeqLM.from_pretrained(args.modelname, from_pt=args.from_pytorch)
    tokenized_datasets, tokenizer = cocitedata.load_dataset(args)

    ### build datasets
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
    tokenized_train = tokenized_datasets["train"]
    tokenized_test = tokenized_datasets["test"]

    if args.debug:
        tokenized_train = tokenized_train.select(list(range(2)))
        tokenized_test = tokenized_test.select(list(range(2)))

    training_columns = ["attention_mask", "input_ids", "labels", "label_occurrences"]
    tf_train_set = tokenized_train.to_tf_dataset(
        columns=training_columns,
        shuffle=True,
        batch_size=args.batchsize,
        collate_fn=data_collator,
    )

    tf_test_set = tokenized_test.to_tf_dataset(
        columns=training_columns,
        shuffle=True,
        batch_size=args.batchsize,
        collate_fn=data_collator,
    )

    # valid number between 100 and 5000
    ds_len = len(tokenized_test["label"])
    num_demo_samples = max(100, ds_len // 20)
    num_demo_samples = min(10000, num_demo_samples)

    if num_demo_samples > ds_len:
        num_demo_samples = ds_len

    wandb.summary.update({"num_demo_samples": num_demo_samples})

    generation_test_dataset = (
        tokenized_test
        .shuffle(seed=42)
        .select(list(range(num_demo_samples)))
        .to_tf_dataset(
            batch_size=args.eval_batchsize,
            drop_remainder=True,
            columns=["input_ids", "attention_mask", "labels", "label_occurrences"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )
    generation_train_dataset = (
        tokenized_train
        .shuffle(seed=42)
        .select(list(range(num_demo_samples)))
        .to_tf_dataset(
            batch_size=args.batchsize,
            drop_remainder=True,
            columns=["input_ids", "attention_mask", "labels", "label_occurrences"],
            shuffle=False,
            collate_fn=data_collator,
        )
    )
    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)  # TODO warning high lr
    model.compile(optimizer=optimizer)

    top_ks = [1, 3, 5, 10, 20]
    max_k = max(top_ks)
    max_k = min(max_k, args.topk) 
    top_ks = [k for k in top_ks if k <= max_k]

    ### build callbacks
    metric_test_callback = callbacks.KerasMetricCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=generation_test_dataset, prefix="test_",
        args=args, batch_size=args.batchsize,
        len_train_dataset = len(tokenized_train["label"]),
        top_ks=top_ks,
    )
    metric_train_callback = callbacks.KerasMetricCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=generation_train_dataset, prefix="train_",
        args=args, batch_size=args.batchsize,
        len_train_dataset = len(tokenized_train["label"]),
        top_ks=top_ks,
    )
    wandb_callback = WandbCallback(save_model=not args.debug)

    callbacks_list = [
        wandb_callback,
        metric_test_callback, 
        metric_train_callback,
    ]

    if not args.debug:
        save_model_callback = SaveModelCallback(
            model=model,
            tokenizer=tokenizer,
            args=args,
            len_train_dataset = len(tokenized_train["label"]),
        )
        callbacks_list.append(save_model_callback)

    ### train model
    if not args.notraining:
        if not args.debug and args.co2_tracking:
            tracker = EmissionsTracker()
            tracker.start()

        model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=args.epochs, callbacks=callbacks_list)

        if not args.debug and args.co2_tracking:
            co2_emissions = tracker.stop()
            wandb.log({"CO2_emissions (in Kg)": co2_emissions})

    ### evaluate model
    if not args.noevaluation:
        train_helpers.evaluate(
            model, tf_test_set,
            prefix="test_", args=args,
            top_ks=top_ks, tokenizer=tokenizer)
        train_helpers.evaluate(
            model, generation_train_dataset,
            prefix="train_", args=args,
            top_ks=top_ks, tokenizer=tokenizer)

if __name__ == "__main__":
    main()