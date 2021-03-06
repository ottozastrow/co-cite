import logging
import os
import re
from time import sleep
from typing import Callable, List, Optional, Union

import numpy as np
from sqlalchemy import all_
import tensorflow as tf
from packaging.version import parse
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import wandb

import train_helpers

logger = logging.getLogger(__name__)


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, tokenizer, args, len_train_dataset,training_step=0):
        self.model = model
        self.tokenizer = tokenizer
        self.training_step = training_step
        self.epochcounter = 0
        self.args = args

        if args.debug:
            self.log_interval = (len_train_dataset / args.batchsize) // 2
            self.log_interval = max(1, self.log_interval)
        else:
            # schedule logging 5 times per epoch
            self.log_interval = (len_train_dataset / args.batchsize) // 5

        self.save_path = "../model_save/" + args.modelname + "_" + str(wandb.run.id) + "/"
        self.save_path += "debug/" if args.debug else ""
        self.training_step_from_filepath()

    def training_step_from_filepath(self):
        self.training_step = 0
        # when continuning training from a checkpoint set to non zero.
        if "model_save" in self.args.modelname:
            # load local checkpoint instead of huggingface hub model
            # set number of train steps if possible
            if "_step_" in self.args.modelname:
                # find first int in string after substring "_steps_"
                steps_text = self.args.modelname.split("_step_")[-1]
                self.training_step = int(re.search(r'\d+', steps_text).group())

    def on_epoch_end(self, epoch, logs=None):
        self.save_model(epoch)
        self.epochcounter += 1

    def on_train_batch_end(self, batch, logs=None):
        if self.training_step % self.log_interval == 0:
            self.save_model(self.epochcounter)

        self.training_step += 1

    def save_model(self, epoch):
        if not self.args.debug:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            name = f"{self.save_path}epoch_{epoch}_step_{self.training_step * self.args.batchsize}"

            wandb.log({"last_saved_model": name})
            self.model.save_pretrained(name)
            self.tokenizer.save_pretrained(name)
            print(f"\nSaved model and toknizer to {name}")


class KerasMetricCallback(Callback):
    """
    copied from transformers library
    """
    def __init__(
        self,
        eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict],
        tokenizer,
        model,
        args,
        top_ks: List[int],
        output_cols: Optional[List[str]] = None,
        label_cols: Optional[List[str]] = None,
        batch_size: int = 1,
        prefix: Optional[str] = None,
        len_train_dataset: int = 0,
    ):
        super().__init__()
        self.step_num = 1
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.prefix = prefix
        self.args = args
        self.top_ks = top_ks, 
        self.label_cols = label_cols
        if not isinstance(eval_dataset, tf.data.Dataset):
            if batch_size is None:
                raise ValueError(
                    "When passing data to KerasMetricCallback that is not a pre-batched tf.data.Dataset "
                    "the batch_size argument must be set."
                )
            # Wrap a tf.data.Dataset around it
            eval_dataset = tf.data.Dataset.from_tensor_slices(eval_dataset).batch(batch_size, drop_remainder=False)
        self.eval_dataset = eval_dataset
        if args.debug:
            self.log_interval = (len_train_dataset / batch_size) // 2 
            self.log_interval = max(1, self.log_interval)
        else:
            if args.evaluations_per_epoch == 0:
                self.log_interval = -1 
            else:
                self.log_interval = (len_train_dataset / batch_size) // args.evaluations_per_epoch
        print("log interval", self.log_interval, "train size", len_train_dataset, "batch size", batch_size)

        self.output_cols = output_cols
        # This next block attempts to parse out which elements of the dataset should be appended to the labels list
        # that is passed to the metric_fn
        if isinstance(eval_dataset.element_spec, tuple) and len(eval_dataset.element_spec) == 2:
            input_spec, label_spec = eval_dataset.element_spec
        else:
            input_spec = eval_dataset.element_spec
            label_spec = None
        if label_cols is not None:
            for label in label_cols:
                if label not in input_spec:
                    raise ValueError(f"Label {label} is in label_cols but could not be found in the dataset inputs!")
            self.label_cols = label_cols
            self.use_keras_label = False
        elif label_spec is not None:
            # If the dataset inputs are split into a 2-tuple of inputs and labels,
            # assume the second element is the labels
            self.label_cols = None
            self.use_keras_label = True
        elif "labels" in input_spec:
            self.label_cols = ["labels"]
            self.use_keras_label = False
            logging.warning("No label_cols specified for KerasMetricCallback, assuming you want the 'labels' key.")
        elif "start_positions" in input_spec and "end_positions" in input_spec:
            self.label_cols = ["start_positions", "end_positions"]
            self.use_keras_label = False
            logging.warning(
                "No label_cols specified for KerasMetricCallback, assuming you want the "
                "start_positions and end_positions keys."
            )
        else:
            raise ValueError("Could not autodetect label_cols for KerasMetricCallback, please specify them!")
        if parse(tf.__version__) < parse("2.7"):
            logging.warning("TF versions less than 2.7 may encounter issues with KerasMetricCallback!")

    @staticmethod
    def _concatenate_batches(batches, padding_index=-100):
        # If all batches are unidimensional or same length, do a simple concatenation
        if batches[0].ndim == 1 or all([batch.shape[1] == batches[0].shape[1] for batch in batches]):
            return np.concatenate(batches, axis=0)

        # Welp, they're not the same length. Let's do some padding
        max_len = max([batch.shape[1] for batch in batches])
        num_samples = sum([batch.shape[0] for batch in batches])
        output = np.full_like(
            batches[0],
            fill_value=padding_index,
            shape=[num_samples, max_len] + list(batches[0].shape[2:])
        )
        # i keeps track of which part of the
        # concatenated array we're writing the next batch to
        i = 0
        for batch in batches:
            output[i: i + len(batch), : batch.shape[1]] = batch
            i += len(batch)
        return output

    def _postprocess_predictions_or_labels(self, inputs):
        if isinstance(inputs[0], dict):
            outputs = dict()
            for key in inputs[0].keys():
                outputs[key] = self._concatenate_batches([batch[key] for batch in inputs])
            # If it's a dict with only one key, just return the array
            if len(outputs) == 1:
                outputs = list(outputs.values())[0]
        elif isinstance(inputs[0], list) or isinstance(inputs[0], tuple):
            outputs = []
            for input_list in zip(*inputs):
                outputs.append(self._concatenate_batches(input_list))
            if len(outputs) == 1:
                outputs = outputs[0]  # If it's a list with only one element, just return the array
        elif isinstance(inputs[0], np.ndarray):
            outputs = self._concatenate_batches(inputs)
        elif isinstance(inputs[0], tf.Tensor):
            outputs = self._concatenate_batches([tensor.numpy() for tensor in inputs])
        else:
            raise TypeError(f"Couldn't handle batch of type {type(inputs[0])}!")
        return outputs

    def on_epoch_end(self, epoch, logs=None):
        train_helpers.evaluate(
            self.model, self.eval_dataset,
            prefix=self.prefix, args=self.args, top_ks=self.top_ks,
            tokenizer=self.tokenizer)

    def on_train_batch_end(self, batch, logs=None):
        if not self.args.evaluations_per_epoch == 0:
            if self.step_num % self.log_interval == 0 and self.prefix != "train_":
                self.on_epoch_end(0, logs=logs)
            self.step_num += 1

