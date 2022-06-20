import argparse


def cmd_arguments(debug=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", help="name on huggingface model hub", default="t5-small")
    parser.add_argument("--wandb_mode", help="online, offline or disabled mode for wandb logging", default="online")
    parser.add_argument("--epochs", type=int, help="number of epochs", default=2)
    parser.add_argument("--batchsize", type=int, help="batch size", default=8)
    parser.add_argument("--contextlength", type=int, help="context length", default=1200)
    parser.add_argument("--input_tokens", type=int, help="input token length", default=256)
    parser.add_argument("--output_tokens", type=int, help="output token length", default=42)
    parser.add_argument("--topk", type=int, help="top k for beam search and accuracy", default=3)
    parser.add_argument("--miniature_dataset", 
                        help="for debugging only use 20 samples", action="store_true")
    parser.add_argument("--notraining", help="skip training pipeline", action="store_true")
    parser.add_argument("--noevaluation", help="skip post training evaluation pipeline", action="store_true")
    parser.add_argument("--rebuild_dataset", help="instead of attempting to load existing datasets rebuild it", action="store_true")
    parser.add_argument("--debug", 
                        help="make data and model tiny for fast local debugging", action="store_true")
    parser.add_argument("--miniature_dataset_size", help="max number of documents to use for building dataset", 
                        type=int, default=10)
    parser.add_argument("--data_dir", help="directory where data is stored",
                        default="../../external_projects/bva-citation-prediction/data/preprocessed-cached/preprocessed-cached-v4/")
    args = parser.parse_args()

    
    if args.debug or debug:
        args.miniature_dataset = True
        args.miniature_dataset_size = 2
        args.batchsize = 1
        # args.input_tokens=8
        # args.output_tokens=4
    
        # args.wandb_mode = "disabled"
    return args