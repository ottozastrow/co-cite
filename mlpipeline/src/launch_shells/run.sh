bsub -W 700 -n 1 -R "rusage[mem=22800,ngpus_excl_p=1]" python train.py --topk=3 --samples=1000 --notraining --tags=eval_speed --runname=beamsearch_top3

bsub -W 700 -n 1 -R "rusage[mem=22800,ngpus_excl_p=1]" python train.py --topk=1 --samples=1000 --notraining --tags=eval_speed --runname=greedysearch_top1

bsub -W 700 -n 1 -R "rusage[mem=22800,ngpus_excl_p=1]" python train.py --topk=3 --samples=1000 --notraining --tags=eval_speed --runname=sampling_top3 --sample_decoding=True
