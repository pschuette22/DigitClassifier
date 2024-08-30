
# Update the pip requirements file
update-requirements:
	pip freeze -l > requirements.txt

font-dataset:
	python digitclassifier/dataset/builder.py fonts/

garden-dataset:
	python digitclassifier/dataset/garden.py -m $(filter-out $@,$(MAKECMDGOALS))

models:
	python digitclassifier/train.py

compare:
	python digitclassifier/compare.py -m $(filter-out $@,$(MAKECMDGOALS))

# Prevent make from interpreting the arguments as targets
%:
    @: