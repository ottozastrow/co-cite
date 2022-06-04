import numpy as np
import pdb

from datasets import load_metric
import wandb

def create_metrics(tokenizer):
    """Function that is passed to Trainer to compute training metrics"""

    metric_bleu = load_metric("bleu")

    def compute_metrics(eval_preds):
        logits = eval_preds.predictions
        labels = eval_preds.label_ids
        # logits[0] are the outputs batch x seq_len x vocab_size
        # logits[1] are (TODO check) output attention masks batch input_seq_len x 512
        # labels is 

        ### bleu metric
        predictions = np.argmax(logits[0], axis=-1)
        results = metric_bleu.compute(predictions=[predictions], references=[labels])

        ### sequence level accuracy
        # first element wise comparison, if there is a single false value, then the whole sequence is wrong
        sample_wise_acc = np.equal(predictions, labels).all(axis=1)
        results["accuracy"] = np.mean(sample_wise_acc)

        ### sample outputs
        num_demo_samples = 5
        sample_outputs = tokenizer.batch_decode(
            predictions[:num_demo_samples], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)
        # outputs = [output.replace("@cit@", "") for output in outputs]
        sample_labels = tokenizer.batch_decode(
            labels[:num_demo_samples], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)

        results["sample_predictions"] = sample_outputs
        results["sample_groundtruths"] = sample_labels

        my_table = wandb.Table(columns=["prediction", "groundtruth"], 
        data=[list(t) for t in zip(sample_outputs, sample_labels)])
        wandb.log({"demo": my_table})
        results["samples"] = my_table
        return results
    return compute_metrics


def create_tokenize_function(tokenizer):
    """Mapping function that tokanizes all relevant dataset entries."""
    def tokenize_function(examples):
            inputs = [input for input in examples['text']]
            targets = [target for target in examples['label']]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding=True)

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=32, truncation=True, padding=True)

            model_inputs["labels"] = labels["input_ids"]
            # TODO remove unused columns here?
            return model_inputs
    return tokenize_function