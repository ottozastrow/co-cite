import numpy as np
import pdb

from datasets import load_metric
import wandb

from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    BeamSearchScorer
)
import pdb
# def logits_to_topk(logits, model, args):
#     # input_ids = torch.ones((args.topk, 1), device=model.device, dtype=tf.)

#     logits_processor = LogitsProcessorList(
#         [MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),]
#     )
#     model_kwargs = {
#         "encoder_outputs": model.get_encoder()(
#             logits.repeat_interleave(args.topk, dim=0), return_dict=True
#     )}

#     beam_scorer = BeamSearchScorer(
#         batch_size=args.batchsize,
#         num_beams=args.topk,
#         #device=model.device,
#     )
#     outputs = model.beam_search(logits, beam_scorer, logits_processor=logits_processor)
#     return outputs


def batch_accuracy(predictions, labels):
    return np.mean([pred == label for pred, label in list(zip(predictions, labels))])


def create_metrics_fn(prefix, tokenizer, model, args):
    """Function that computes all metrics for the given dataset.
    Args:
        prefix: name prefix for the metrics (e.g. "test_")
        tokenizer: tokenizer to use for the metrics
    Returns:
        results: dictionary with all metrics
        logs to wandb
    """

    rouge_metric = load_metric("rouge")
    import timeit
    def metrics_fn(data, topk=1):
        # predictions, labels = data
        predictions = data[0]["sequences"]
        labels = data[1]
        scores = data[0]["scores"]
        
        # use transformers beam search to get the top k predictions
        # predictions = logits_to_topk(logits, model, args)
        print("finished generating predictions")
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        print("finished decoding predictions")
        # decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print("finished decoding labels")

        ### Compute ROUGE
        # measure time with timeit
        results = {}
        if topk == 1:
            result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
            start = timeit.default_timer()
            end = timeit.default_timer()
            print("rouge time: ", end - start)
            results =  {(prefix + key): value.mid.fmeasure * 100 for key, value in result.items()}
            wandb.log({prefix + "rouge": results})

        ### accuracy
        if topk != 1:
            accuracy_per_sample = []
            for i in range(topk):
                accuracy_per_sample.append(batch_accuracy(decoded_predictions[i], decoded_labels))
            
            results[prefix + 'acc_top' + str(topk)] = np.mean(np.array(accuracy_per_sample))
            wandb.log({prefix + 'acc_top' + str(topk): results[prefix + 'acc_top' + str(topk)]})
        
        results[prefix + 'acc'] = batch_accuracy(decoded_predictions[0], decoded_labels)
    
        wandb.log({prefix + "accuracy": results[prefix + 'acc']})

        ### sample outputs
        my_table = wandb.Table(columns=["prediction", "groundtruth"], 
        data=[list(t) for t in zip(decoded_predictions, decoded_labels)])
        wandb.log({prefix + "demo": my_table})

        return results
    return metrics_fn

# def create_metrics(tokenizer):
#     """Function that is passed to Trainer to compute training metrics"""

#     metric_bleu = load_metric("bleu")

#     def compute_metrics(eval_preds):
#         logits = eval_preds.predictions
#         labels = eval_preds.label_ids
#         # logits[0] are the outputs batch x seq_len x vocab_size
#         # logits[1] are (TODO check) output attention masks batch input_seq_len x 512
#         # labels is 

#         ### bleu metric
#         predictions = np.argmax(logits[0], axis=-1)
#         results = metric_bleu.compute(predictions=[predictions], references=[labels])

#         ### sequence level accuracy
#         # first element wise comparison, if there is a single false value, then the whole sequence is wrong
#         sample_wise_acc = np.equal(predictions, labels).all(axis=1)
#         results["accuracy"] = np.mean(sample_wise_acc)

#         ### sample outputs
#         num_demo_samples = 5
#         sample_outputs = tokenizer.batch_decode(
#             predictions[:num_demo_samples], 
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True)
#         # outputs = [output.replace("@cit@", "") for output in outputs]
#         sample_labels = tokenizer.batch_decode(
#             labels[:num_demo_samples], 
#             skip_special_tokens=True,
#             clean_up_tokenization_spaces=True)

#         results["sample_predictions"] = sample_outputs
#         results["sample_groundtruths"] = sample_labels

#         my_table = wandb.Table(columns=["prediction", "groundtruth"], 
#         data=[list(t) for t in zip(sample_outputs, sample_labels)])
#         wandb.log({"demo": my_table})
#         results["samples"] = my_table
#         return results
#     return compute_metrics


def create_tokenize_function(tokenizer):
    """Mapping function that tokanizes all relevant dataset entries."""
    def tokenize_function(examples):
            inputs = [input for input in examples['text']]
            targets = [target for target in examples['label']]
            model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=32, truncation=True, padding="max_length")

            model_inputs["labels"] = labels["input_ids"]
            # TODO remove unused columns here?
            return model_inputs
    return tokenize_function