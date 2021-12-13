install:
	pip install -r requirements.txt

collect-tweets:
	cd src && python3 -i data_collection.py

train-model0:
	cd src && python3 -i model0_wordbased.py --embedding_dim 64 --gru_dim 64 --dense_dim 64 --batch_size 256 --learning_rate 0.0001