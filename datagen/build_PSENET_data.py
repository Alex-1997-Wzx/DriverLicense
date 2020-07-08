''' 将data_generator.py生成的数据根据PSENET要求的目录格式摆放 '''
import os
import random
import shutil
from tqdm import tqdm

chs = 'abcdefghijklmnopqrstwvuxyzABCDEFGHIJKLMNOPQRSTWVUXYZ'


history_file = './gen/psenet_history_20191129.txt'
root_image_dir = './gen/original'
root_label_dir = './gen/labels_ctw'
train_dir = './gen/PSENET/train'
test_dir = './gen/PSENET/test'

train_image_dir = os.path.join(train_dir, 'text_image')
train_label_dir = os.path.join(train_dir, 'text_label_curve')
test_image_dir = os.path.join(test_dir, 'text_image')
test_label_dir = os.path.join(test_dir, 'text_label_curve')

for d in [train_dir, test_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
        print('make dir: ', d)


def move_origin_to_psenet():
    history = []  # 记录对应名称: 旧名称,新名称

    image_files = os.listdir(root_image_dir)

    random.shuffle(image_files)

    train_len = int(len(image_files) * 0.9)
    # test_len = len(image_files) - train_len

    train_images = image_files[:train_len]
    test_images = image_files[train_len:]

    for f in tqdm(train_images):
        header = ''.join(random.sample(chs, 4))
        new_f = header + f.split('_')[1]
        history.append(f+','+new_f)
        # move image
        root = os.path.join(root_image_dir, f)
        target = os.path.join(train_image_dir, new_f)
        shutil.move(root, target)
        # move label
        root = os.path.join(root_label_dir, f[:-4]+'.txt')
        target = os.path.join(train_label_dir, new_f[:-4]+'.txt')
        shutil.move(root, target)
    for f in tqdm(test_images):
        header = ''.join(random.sample(chs, 4))
        new_f = header + f.split('_')[1]
        history.append(f+','+new_f)
        # move images
        root = os.path.join(root_image_dir, f)
        target = os.path.join(test_image_dir, new_f)
        shutil.move(root, target)
        # move label
        root = os.path.join(root_label_dir, f[:-4]+'.txt')
        target = os.path.join(test_label_dir, new_f[:-4]+'.txt')
        shutil.move(root, target)

    with open(history_file, 'w', encoding='utf-8') as hf:
        hf.write('\n'.join(history))

# def move_back():
#     for path in [train_image_dir, test_image_dir]:
#         for f in tqdm(os.listdir(path)):
#             fn = os.path.join(path, f)
#             org = os.path.join(root_image_dir, f)
#             shutil.move(fn, org)
#     for path in [train_label_dir, test_label_dir]:
#         for f in tqdm(os.listdir(path)):
#             fn = os.path.join(path, f)
#             org = os.path.join(root_label_dir, f)
#             shutil.move(fn, org)


if __name__ == '__main__':
    move_origin_to_psenet()
    # move_back()