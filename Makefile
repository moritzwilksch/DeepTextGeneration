install:
	pip install -r requirements.txt

format:
	black .

collect-tweets:
	cd src && python3 -i data_collection.py

train-word:
	python3 -i src/modeling/model0_wordbased.py 
	
train-char:
	python3 -i src/modeling/model1_charbased.py

generate-word:
	python3 src/generate_from_model.py \
	--mode word

generate-char:
	python3 src/generate_from_model.py \
	--mode char