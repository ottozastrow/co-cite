cd elasticsearch-7.9.2
./bin/elasticsearch &
sleep 120
cd ../
python retrieval/retrieve.py --samples=10000 --retriever="dpr" --rebuild_dataset --batchsize=4 --wandb_mode=disabled
