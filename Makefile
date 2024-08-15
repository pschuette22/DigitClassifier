
# Update the pip requirements file
update_requirements:
	pip freeze -l > requirements.txt

font-dataset:
	python digitclassifier/dataset/builder.py fonts/

models:
	python digitclassifier/train.py