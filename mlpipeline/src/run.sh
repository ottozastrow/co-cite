cd elasticsearch-7.9.2
./bin/elasticsearch &
sleep 30
cd ../
python retrieve.py --samples=10000 --retriever="embedding" --rebuild_dataset
