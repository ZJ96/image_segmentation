import os
import shutil

def check_new_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.mkdir(dir_name)
