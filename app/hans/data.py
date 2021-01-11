import os
import shutil


DATA_DIR = '../../data'


def make_path(basepath, dirname):
    dirpath = os.path.join(basepath, dirname)
    try:
        os.mkdir(dirpath)
    except FileExistsError:
        pass
    return dirpath


def copy_files(origin_dir, target_dir, scope_from, scope_to):
    origin_data = os.listdir(origin_dir)
    for fname in origin_data[scope_from:scope_to]:
        src = os.path.join(origin_dir, fname)
        dst = os.path.join(target_dir, fname)
        try:
            shutil.copyfile(src, dst)
        except shutil.SameFileError:
            pass


def load_data(source_dir, category):
    base_path = make_path(source_dir, category)
    origin_single_default_dir = make_path(base_path, 'Single_Default')

    data_list = os.listdir(origin_single_default_dir)
    data_size = len(data_list)

    scope = {
        'train': (0, int(data_size/100)*16),
        'valid': (int(data_size/100)*16, int(data_size/100)*20),
    }

    train_dir = make_path(DATA_DIR, 'train')
    train_target_dir = make_path(train_dir, category)
    train_single_default_dir = make_path(train_target_dir, 'Single_Default')
    copy_files(
        origin_single_default_dir,
        train_single_default_dir,
        scope['train'][0],
        scope['train'][1]
    )

    valid_dir = make_path(DATA_DIR, 'valid')
    valid_target_dir = make_path(valid_dir, category)
    valid_single_default_dir = make_path(valid_target_dir, 'Single_Default')
    copy_files(
        origin_single_default_dir,
        valid_single_default_dir,
        scope['valid'][0],
        scope['valid'][1]
    )
