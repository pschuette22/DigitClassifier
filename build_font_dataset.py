import os
import argparse
import zipfile
import random
from PIL import Image, ImageDraw, ImageFont


# TODO: consider a trie or an otherwise better datastructure
# Maybe a yml with rules and fonts separated to initialize a model
def load_ignore_ruleset():
    ignore_rules = []
    with open('dataset/ignored.txt') as file:
        for line in file:
            stripped = line.strip()
            # Ignore comments and empty strings
            if stripped.startswith("#") or len(stripped) == 0:
                continue
            print(f"adding ignore rule: {stripped}")
            ignore_rules.append(stripped)
    return ignore_rules

def ignore_font(font_name, ignore_rules):
    for rule in ignore_rules:
        if font_name.count(rule) > 0:
            print(f"ignoring {font_name}")
            return True
    return False

def find_font_vector(font_path, text):
    """
        Find the origin and font size that fits the text in a 28 x 28 image
    """
    font_size = 27
    x,y = 0,0
    iterations = 0
    while True:
        iterations += 1
        if iterations > 20:
            print("failed to find vector from iterations!")
            raise Exception("Failed to find font vector")
        
        font = ImageFont.truetype(font=font_path, size=font_size, layout_engine=ImageFont.Layout.BASIC)
        bounding_box = font.getbbox(text)
        width = bounding_box[2] - bounding_box[0]
        height = bounding_box[3] - bounding_box[1]
        if height == 0:
            # This happens when the glpyh is defined but empty
            raise Exception("Font height is 0, unable to draw glyph")
        # assuming font glyphs have a greater width than height
        if height > 21 or width > 21:
            font_size -= 2
        elif height < 16 and width < 16:
            font_size += 2
        else:
            x = ((28 - width) // 2)
            y = ((28 - height) // 2)
            break    
    return (font_size, (x,y))

def main(fonts_path, start_digit, end_digit):
    print("build from fonts at: ", fonts_path)
    print("lower bound digit: ", start_digit)
    print("upper bound digit: ", end_digit)
    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, 'dataset/fonts')
    fonts_dir = os.path.join(fonts_path, 'ofl')

    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    # Splits
    train = 80
    train_fonts = 0
    test = 10
    test_fonts = 0
    validate = 10
    validate_fonts = 0

    for digit in range(int(start_digit), int(end_digit) + 1):
        os.makedirs(os.path.join(output_dir, "train", str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "test", str(digit)), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "validate", str(digit)), exist_ok=True)

    total_fonts = 0
    ignore_ruleset = load_ignore_ruleset()
    for root, _, files in os.walk(fonts_dir):
        if "__MACOSX" in root:
            continue
        
        for file in files:
            if file.endswith('.ttf'):
                if ignore_font(file, ignore_ruleset):
                    continue
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
                    # 8 bit grayscale https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
                    img = Image.new(mode='L', size=(28, 28), color=background)
                    draw = ImageDraw.Draw(img)
                    try:
                        text = str(digit)
                        font_vector = find_font_vector(font_path, text)
                        font_size = font_vector[0]
                        font_coords = font_vector[1]
                        font = ImageFont.truetype(font=font_path, size=font_size, layout_engine=ImageFont.Layout.BASIC)
                        draw.text((font_coords[0], font_coords[1]), text, fill=fill, font=font, anchor="lt")
                        img.save(os.path.join(result_dir, text, f"{digit}_{font_name}.jpg"))
                    except Exception as error:
                        print(f"!!! Failed to draw {font} ({font_path}) !!! \n{error}")
                        total_fonts -= 1 # Assumes we aren't able to draw any digits and easier than skipping the incriment
                        break

                total_fonts += 1

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