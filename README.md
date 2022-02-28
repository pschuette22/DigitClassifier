# DigitClassifier
Use Google fonts to generate a digit classifying dataset for CoreML


## Setup
[Image Magick](https://imagemagick.org/)
```
brew install imagemagick
```

Download a variety of fonts (the more the better!) from [Google Fonts](https://fonts.google.com/) and add them to the `fonts/` folder

## Run it
Run the generate script and supply it two parameters: lower and upper bound.
This was originally created to build a digit classifier for a Sudoku project, so the 0 was omitted.

```
./generate.sh 1 9
```
