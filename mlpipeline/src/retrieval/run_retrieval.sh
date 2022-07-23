cd elasticsearch-7.9.2
./bin/elasticsearch &
sleep 40
cd ../
python retrieval/retrieve.py --samples=1000 --retriever="dpr" --rebuild_dataset --batchsize=2
