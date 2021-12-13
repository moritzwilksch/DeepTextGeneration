install:
	conda install -y --file requirements.txt

collect-tweets:
	cd src && python3 -i data_collection.py