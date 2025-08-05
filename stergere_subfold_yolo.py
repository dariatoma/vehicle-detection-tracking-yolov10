import os
import shutil

def flatten_all_files(folder_path, allowed_exts):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            ext = os.path.splitext(file)[-1].lower()
            if ext in allowed_exts:
                src = os.path.join(root, file)
                dst = os.path.join(folder_path, file)
                if not os.path.exists(dst):
                    shutil.move(src, dst)

        if root != folder_path:
            try:
                os.rmdir(root)
                print(f"Șters folder gol: {root}")
            except OSError:
                pass

base_dir = '/home/daria/PycharmProjects/proiect_daria/PythonProject/car_dataset'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

flatten_all_files(image_dir, ['.jpg', '.jpeg', '.png'])
flatten_all_files(label_dir, ['.txt'])