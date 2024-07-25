import os
import zipfile
import random
from PIL import Image, ImageDraw, ImageFont

def main(start_digit, end_digit):
    working_dir = os.getcwd()
    output_dir = os.path.join(working_dir, 'dataset')
    fonts_dir = os.path.join(working_dir, 'fonts')

    if os.path.exists(output_dir):
        os.system(f'rm -rf {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    extensions = ["ttf", "otf"]
    font_max = 100

    for digit in range(int(start_digit), int(end_digit) + 1):
        os.makedirs(os.path.join(output_dir, str(digit)), exist_ok=True)

    total_fonts = 0
    entries = 0

    for font_zip in os.listdir(fonts_dir):
        if font_zip.endswith('.zip'):
            font_family = os.path.splitext(font_zip)[0]
            with zipfile.ZipFile(os.path.join(fonts_dir, font_zip), 'r') as zip_ref:
                zip_ref.extractall(os.path.join(fonts_dir, font_family))
            total_fonts += 1

            for ext in extensions:
                for root, _, files in os.walk(os.path.join(fonts_dir, font_family)):
                    if "__MACOSX" in root:
                            continue
                    for file in files:                        
                        if file.endswith(f'.{ext}'):
                            font_path = os.path.join(root, file)
                            font_name = os.path.splitext(file)[0]
                            print(f"Processing font: {font_name}")
                            print(f"Filepath: {font_path}")

                            fill = "white"

                            for digit in range(int(start_digit), int(end_digit) + 1):
                                background = "black"

                                img = Image.new('RGB', (28, 28), color=background)
                                d = ImageDraw.Draw(img)
                                font = ImageFont.truetype(font=font_path, size=18, layout_engine=ImageFont.Layout.BASIC)
                                d.text((14, 14), str(digit), fill=fill, font=font, anchor="mm")
                                img.save(os.path.join(output_dir, str(digit), f"{digit}_{font_name}_original.jpg"))

                            entries += 1

                            if entries >= font_max:
                                break
                    if entries >= font_max:
                        break

            os.system(f'rm -rf {os.path.join(fonts_dir, font_family)}')

    print(f"Total fonts: {total_fonts}")
    print(f"Total entries: {entries}")

if __name__ == "__main__":
    import sys
    start_digit = sys.argv[1]
    end_digit = sys.argv[2]
    main(start_digit, end_digit)