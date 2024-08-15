
# Update the pip requirements file
update-requirements:
	pip freeze -l > requirements.txt

font-dataset:
	python digitclassifier/dataset/builder.py fonts/

garden-dataset:
	python digitclassifier/dataset/garden.py

models:
	python digitclassifier/train.py