import os
import argparse
import zipfile
import random
from PIL import Image, ImageDraw, ImageFont

def main(fonts_path, start_digit, end_digit):
    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, 'dataset')
    fonts_dir = os.path.join(fonts_path, 'ofl')

    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    font_max = 5 # Bail early

    # Splits
    train = 90
    train_fonts = 0
    test = 10
    test_fonts = 0
    validate = 0
    validate_fonts = 0

    for digit in range(int(start_digit), int(end_digit) + 1):
        os.makedirs(os.path.join(output_dir, "train", str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validate", str(digit)), exist_ok=True)

    total_fonts = 0

    for root, _, files in os.walk(fonts_dir):
        if "__MACOSX" in root:
            continue
        if "Barcode" in root:
            continue
        if "Blank" in root:
            continue
        if "Highlight" in root:
            continue

        for file in files:
            if file.endswith('.ttf'):
                # Determine what set this belongs to
                seed = random.randint(0, train + test + validate)
                result_dir = output_dir
                if seed <= train:
                    result_dir=os.path.join(output_dir, "train")
                    train_fonts += 1
                elif seed <= train + test:
                    result_dir=os.path.join(output_dir, "test")
                    test_fonts += 1
                else:
                    result_dir=os.path.join(output_dir, "validate")
                    validate_fonts += 1

                font_path = os.path.join(root, file)
                font_name = os.path.splitext(file)[0]
                print(f"{total_fonts}: Processing font {font_name}")
                fill = "white"

                for digit in range(int(start_digit), int(end_digit) + 1):
                    background = "black"

                    img = Image.new('RGB', (28, 28), color=background)
                    draw = ImageDraw.Draw(img)
                    try:
                        font = ImageFont.truetype(font=font_path, size=18, layout_engine=ImageFont.Layout.BASIC)
                        draw.text((14, 14), str(digit), fill=fill, font=font, anchor="mm")
                        img.save(os.path.join(result_dir, str(digit), f"{digit}_{font_name}.jpg"))
                    except:
                        print(f"!!! Failed to draw {font} ({font_path}) !!!")
                        total_fonts -= 1 # Assumes we aren't able to draw any digits and easier than skipping the incriment
                        break

                total_fonts += 1

                if font_max > 0 & total_fonts >= font_max:
                    break

        if font_max > 0 & total_fonts >= font_max:
            break

    print()
    print(f"Total fonts: {total_fonts}")
    print(f"Train fonts: {train_fonts}")
    print(f"Test fonts: {test_fonts}")
    print(f"Validate fonts: {validate_fonts}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process path to google fonts project, lower bound digit, and upper bound digit')
    parser.add_argument('directory', type=str, help='The path to the google fonts project directory')
    parser.add_argument('lower', type=int, help='Lower digit bound')
    parser.add_argument('upper', type=int, help='Upper digit bound')
    args = parser.parse_args()
    if os.path.isdir(args.directory):
        main(args.directory, args.lower, args.upper)
    else:
        raise FileNotFoundError(f"The directory ${args.directory} doesnt exist: pass a valid path to the google fonts repository.")