install:
	pip install -r requirements.txt

format:
	black .
	
collect-tweets:
	cd src && python3 -i data_collection.py

train-model0:
	python3 -i src/model0_wordbased.py --embedding_dim 64 --gru_dim 64 --dense_dim 64 --batch_size 512 --learning_rate 0.0001

train-model1:
	python3 -i src/model1_charbased.py --embedding_dim 64 --gru_dim 64 --dense_dim 64 --batch_size 512 --learning_rate 0.0001