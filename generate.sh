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

train_split=5
valid_split=1
test_split=1
total_split=$((train_split + valid_split + test_split))


background_colors=("white" "white" "white" "snow1" "snow2" "snow3" "LightSkyBlue1")

for digit in $(seq "$start_digit" "$end_digit"); do
    # mkdir -p ${output}/train/${digit}
    # mkdir -p ${output}/valid/${digit}
    # mkdir -p ${output}/test/${digit}
    mkdir -p ${output}/${digit}
done

total_fonts=0
entries=0

for font_zip in ${fonts_dir}/*.zip; do 
    font_family="$(basename -s .zip $font_zip)"
    unzip $font_zip -d ${fonts_dir}/${font_family}
    total_fonts=$((total_fonts+1))
    
    for ext in "${extensions[@]}"; do
        find ${fonts_dir}/${font_family} -type f -not -path "*__MACOSX*" -name "*.${ext}" | while read font; do
            if [[ ! -f "$font" ]]; then 
                echo "No fonts found for family: $font_family with extension: $ext"
                continue; 
            fi

            font_name="$(basename -s .$ext $font)"
            echo "Processing font: $font_name"
            echo "Filepath: $font"
            # Transformations
            fill="white"

            # split_index=$((entries % total_split))
            # if [ "$split_index" -ge 0 ] && [ "$split_index" -lt "$train_split" ]
            # then 
            #     font_split="train"
            # elif [ "$split_index" -ge "$train_split" ] &&  [ "$split_index" -lt $(( $train_split + $valid_split)) ]
            # then 
            #     font_split="valid"
            # else
            #     font_split="test"
            # fi

            for digit in $(seq "$start_digit" "$end_digit"); do

                background="black" # ${background_colors[ $RANDOM % ${#background_colors[@]} ]}

                magick -background $background \
                        -fill $fill \
                        -font $font \
                        -pointsize 18 \
                        -size 28x28 \
                        -gravity center \
                        label:"$digit" \
                        ${output}/${digit}/${digit}_${font_name}_original.jpg
                        # ${output}/${font_split}/${digit}/${digit}_${font_name}_original.jpg

#                for i in $(seq 1 3); do
#                  
#                    lower=$((i*10))
#                    upper=$((i* 10 + 9))
#                    rotation=$(jot -r 1 "$lower" "$upper")
#                    echo "Rotation $rotation"
#    
#                    convert ${output}/${font_split}/${digit}/${digit}_${font_name}_original.jpg \
#                        -background $background \
#                        -rotate $rotation \
#			-gravity center \
#                        -extent 28x28 \
#                        ${output}/${font_split}/${digit}/${digit}_${font_name}_rotated${rotation}.jpg
#                    
#                    reversed=$((-1 * rotation))
#                    convert ${output}/${font_split}/${digit}/${digit}_${font_name}_original.jpg \
#                        -background $background \
#                        -rotate $reversed \
#                        -gravity center \
#			-extent 28x28 \
#                        ${output}/${font_split}/${digit}/${digit}_${font_name}_rotated${reversed}.jpg
#
#                done
            done

            entries=$((entries+1))

            # Uncomment to stop program from overfitting a particularly robust font style
            # if [ $entries -ge $font_max ]; then break 2; fi
        done
    done

    rm -rf ${fonts_dir}/${font_family}
done

echo "Total fonts: $total_fonts"
echo "Total entries: $entries"