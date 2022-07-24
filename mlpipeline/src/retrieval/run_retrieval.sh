cd elasticsearch-7.9.2
./bin/elasticsearch &
sleep 40
cd ../
python retrieval/retrieve.py --retriever="dpr" --batchsize=4 --epochs=5 --retriever_saved_models=../../model_save/retrieval/data_len_all/dpr/ --notraining
