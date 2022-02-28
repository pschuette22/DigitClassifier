#!/bin/bash

start_digit=$1
end_digit=$2

working_dir=$(pwd)
output=${working_dir}/dataset
fonts_dir=${working_dir}/fonts
rm -rf $output
mkdir $output
extensions=("ttf" "otf")
font_max=100

train_split=7
valid_split=1
test_split=2
total_split=$((train_split + valid_split + test_split))


background_colors=("white" "white" "white" "snow1" "snow2" "snow3" "LightSkyBlue1")

for digit in $(seq "$start_digit" "$end_digit"); do
    mkdir -p ${output}/train/${digit}
    mkdir -p ${output}/valid/${digit}
    mkdir -p ${output}/test/${digit}
done

total_fonts=0
entries=0

for font_zip in ${fonts_dir}/*.zip; do 
    font_family="$(basename -s .zip $font_zip)"
    unzip $font_zip -d ${fonts_dir}/${font_family}
    total_fonts=$((total_fonts+1))
    
    for ext in "${extensions[@]}"; do
        for font in ${fonts_dir}/${font_family}/**.${ext}; do
            if [[ ! -f "$font" ]]; then continue; fi

            font_name="$(basename -s .$ext $font)"

            # Transformations
            fill="black"

            split_index=$((entries % total_split))
            if [ "$split_index" -ge 0 ] && [ "$split_index" -lt "$train_split" ]
            then 
                font_split="train"
            elif [ "$split_index" -ge "$train_split" ] &&  [ "$split_index" -lt $(( $train_split + $valid_split)) ]
            then 
                font_split="valid"
            else
                font_split="test"
            fi

            for digit in $(seq 0 9); do

                background=${background_colors[ $RANDOM % ${#background_colors[@]} ]}

                convert -background $background \
                        -fill $fill \
                        -font $font \
                        -pointsize 300 \
                        -extent 416x416 \
                        -gravity center \
                        label:"$digit" \
                        ${output}/${font_split}/${digit}/${digit}_${font_name}.jpg

                # convert ${output}/${font_split}/${digit}/${digit}_${font_name}.jpg \
                #         -borderColor black \
                #         -border 8 \
                #         ${output}/${font_split}/${digit}/${digit}_${font_name}.jpg


                for i in $(seq 1 5); do
                  
                    lower=$((i*5))
                    upper=$((i* 5 + 4))
                    rotation=$(jot -r 1 "$lower" "$upper")
                    echo "Rotation $rotation"
    
                    convert ${output}/${font_split}/${digit}/${digit}_${font_name}.jpg \
                        -background $background \
                        -rotate $rotation \
                        ${output}/${font_split}/${digit}/${digit}_${font_name}_rotated${rotation}.jpg
                    
                    reversed=$((-1 * rotation))
                    convert ${output}/${font_split}/${digit}/${digit}_${font_name}.jpg \
                        -background $background \
                        -rotate $reversed \
                        ${output}/${font_split}/${digit}/${digit}_${font_name}_rotated${rotation}.jpg

                done
            done

            entries=$((entries+1))

            # if [ $entries -ge $font_max ]; then break 2; fi
        done
    done

    rm -rf ${fonts_dir}/${font_family}
done
