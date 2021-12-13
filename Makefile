install:
	conda install -y --file requirements.txt

collect-tweets:
	cd src && python3 data_collection.py