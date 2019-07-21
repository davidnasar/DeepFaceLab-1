from pathlib import Path
from os import scandir

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def get_image_paths(dir_path, image_extensions=image_extensions):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append(x.path)
    return result

def get_image_unique_filestem_paths(dir_path, verbose_print_func=None):
    result = get_image_paths(dir_path)
    result_dup = set()

    for f in result[:]:
        f_stem = Path(f).stem
        if f_stem in result_dup:
            result.remove(f)
            if verbose_print_func is not None:
                verbose_print_func ("Duplicate filenames are not allowed, skipping: %s" % Path(f).name )
            continue
        result_dup.add(f_stem)

    return result
    
def get_file_paths(dir_path):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():
        return [ x.path for x in list(scandir(str(dir_path))) if x.is_file() ]
    return result
    
def get_all_dir_names (dir_path):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():
        return [ x.name for x in list(scandir(str(dir_path))) if x.is_dir() ]
        
    return result
    
def get_all_dir_names_startswith (dir_path, startswith):
    dir_path = Path (dir_path)
    startswith = startswith.lower()

    result = []
    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if x.name.lower().startswith(startswith):
                result.append ( x.name[len(startswith):] )
    return result

def get_first_file_by_stem (dir_path, stem, exts=None):
    dir_path = Path (dir_path)
    stem = stem.lower()

    if dir_path.exists():
        for x in list(scandir(str(dir_path))):
            if not x.is_file():
                continue
            xp = Path(x.path)
            if xp.stem.lower() == stem and (exts is None or xp.suffix.lower() in exts):
                return xp

    return None

def move_all_files (src_dir_path, dst_dir_path):
    paths = get_file_paths(src_dir_path)
    for p in paths:
        p = Path(p)
        p.rename ( Path(dst_dir_path) / p.name )
        
def delete_all_files (dir_path):
    paths = get_file_paths(dir_path)
    for p in paths:
        p = Path(p)
        p.unlink()