import os
import shutil


DATA_DIR = 'data/'


def copy_data(source_dir):
    for category in os.listdir(source_dir):
        split_valid(source_dir, category)


def split_valid(base_path, category, split_rate=0.80):
    source_path = os.path.join(base_path, category)

    for data_dir in os.listdir(source_path):
        data_path = os.path.join(source_path, data_dir)
        data_list = os.listdir(data_path)
        data_size = len(data_list)
        print(f"{category}, size : {data_size}")

        scope = {
            'train': (0, int(data_size * split_rate)),
            'valid': (int(data_size * split_rate), data_size)
        }

        for dir_type in ['train', 'valid']:
            copy_dir = make_path(DATA_DIR, dir_type)
            copy_category_dir = make_path(copy_dir, category)
            copy_data_dir = make_path(copy_category_dir, data_dir)
            copy_files(
                data_path,
                copy_data_dir,
                scope[dir_type][0],
                scope[dir_type][1]
            )
            print(f"{dir_type} : {scope[dir_type][0]} - {scope[dir_type][1]}")


def copy_files(origin_dir, target_dir, scope_from, scope_to):
    origin_data = os.listdir(origin_dir)
    for fname in origin_data[scope_from:scope_to]:
        src = os.path.join(origin_dir, fname)
        dst = os.path.join(target_dir, fname)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass


def make_path(basepath, dirname):
    dirpath = os.path.join(basepath, dirname)
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        pass
    return dirpath
