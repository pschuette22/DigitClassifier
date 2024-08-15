from pathlib import Path

def font_digit(file_path) -> int:
    """Return the degit depicted in an image given it's path."""
    return int(file_path.split('/')[-2])

def font_family(file_path) -> str:
    """Return the font family of an image given it's path."""
    return Path(file_path).stem

# TODO: garden with a seto 