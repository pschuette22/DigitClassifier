# DigitClassifier
Use font files to augment the MNIST Dataset and to train a classifier better suited for printed fonts. This is meant to improve the accuracy of the [SudokuSolver](https://github.com/pschuette22/SudokuSolver) decoding task.

<img src="resources/SudokuSolverDemo.gif" height="300"/>

## Setup
### Install Fonts
Download a variety of fonts (the more the better!) from [Google Fonts](https://fonts.google.com/) or from the [Google Fonts Github Project](https://github.com/google/fonts) and add them to the `fonts/` folder.

Additionally, installing fonts using [fnt](https://github.com/alexmyczko/fnt) will add more training data.

### Setup the project environment
Download and install [miniconda](https://docs.anaconda.com/miniconda/).

Setup the environment. Due to framework conflicts, I recommend using these provided requirements document.

Create the conda environment, activate, install the requirements, and start the notebook.

```bash
conda create -n classifier-env python=3.11 pip
conda activate classifier-env
pip install -r requirements.txt
jupyter activate
```

Open `evaluate.ipbny` in the Juypter notebook run the program from there. 