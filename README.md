# DigitClassifier
Use font files to augment the MNIST Dataset and to train a classifier better suited for printed fonts. This is meant to improve the accuracy of the [SudokuSolver](https://github.com/pschuette22/SudokuSolver) decoding task.

<img src="resources/SudokuSolverDemo.gif" height="300"/>

## Setup
### Install Fonts
Download a variety of fonts (the more the better!) from [Google Fonts](https://fonts.google.com/) or from the [Google Fonts Github Project](https://github.com/google/fonts) and add them to the `fonts/` folder.

Additionally, installing fonts using [fnt](https://github.com/alexmyczko/fnt) will add more training data.

Download ~4200 font files (~1600 fonts), taking up 1.4G of disk space.
```
fnt update
for a in $(fnt search |grep ^google- |sed s,google-,,); do fnt install $a; done
```


### Create environment
Download and install [miniconda](https://docs.anaconda.com/miniconda/).

Create the conda environment, activate, install the requirements, and start the notebook.

```bash
conda create -n classifier-env python=3.11 pip
conda activate classifier-env
pip install -r requirements.txt
```

## Prepare the Dataset
The first step is to prepare the dataset from a set of fonts. Do this by running the build dataset python script over the fonts added to the `fonts` directory

The `digitclassifier/dataset/builder.py` script will convert all fonts found in the `[project}/fonts` directory as well as fnt downloaded fonts into 28x28 grayscale images of each digit. It will ignore fonts with a name partially matching patterns specified in the `dataset/ignored.txt` file.

Font files containing a valid glyph and not matching rules found in the `dataset/ignored.txt` file will be added to the output dataset.

```
make font-dataset
```
<img src="resources/make-font-dataset-short.gif" height="500"/>

This creates structured output in the dataset directory:
```
dataset/
  fonts/
    test/
      0/
        FontName.png
        ...
      9/
        FontName.png
    train/
      ...
    validate/
      ...
```
If all Google and fnt fonts are used, this will produce 10 images of digits using approximately 8200 unique fonts. 


## Train the Models
Once the dataset is created, train the models.
```
make models
```



## Comparing the Models


## Gardening the fonts