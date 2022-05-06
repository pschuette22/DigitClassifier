#!/bin/bash

#
# Copied from https://graphicdesign.stackexchange.com/questions/8422/how-can-i-pixelate-an-image-via-the-command-line-on-linux
#

AMOUNT=$(echo "1.001 - $1" | bc -l)
INFILE=$2
OUFILE=$3

COEFF1=$(echo "100 * $AMOUNT" | bc -l)
COEFF2=$(echo "100 / $AMOUNT" | bc -l)

convert -scale $COEFF1% -scale $COEFF2% $INFILE $OUFILE

