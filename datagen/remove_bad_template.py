''' 移除坏的模板 '''
import os


root_dir = './material/templates'
file_to_remove = []
for f in os.listdir(root_dir):
    tmp_fn = os.path.splitext(f)[0]
    if len(tmp_fn.split('-')) != 3:
        print(f)
        fn = os.path.join(root_dir, f)
        file_to_remove.append(fn)

for f in file_to_remove:
    os.remove(f)