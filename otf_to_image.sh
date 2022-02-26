#!/bin/bash

font_file_path=$1
font_name="$(basename -s .ttf $font_file_path)"
digit=$2

convert -background white -fill black -font $font_file_path -pointsize 300 label:"$digit" ${digit}_${font_name}.png

# convert xc:none[350x350] -background none -fill black -font $font_file -pointsize 300 \
#          -gravity center -annotate 0 "$digit" miff:- | \
# 	convert -dispose background -delay 20 miff:- ${digit}_${font_file}.jpg



