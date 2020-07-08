''' 根据cut的词条图片及图片名称，生成一个CRNN训练数据集（txt文件），
    具体实现：根据[图片名,label]的格式逐行写入到一个txt文件中 '''

import os
import shutil
import random


root_path = './gen/cut'


files = os.listdir(root_path)
len_files = len(files)
print('found files: ', len(files))


rd = list('qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890')

texts = []

# 定时备份, 防止中途死机。。。
copy_texts = []

for i, f in enumerate(files):
    label, suffix = f[:-4].split('_')

    # new_label = pinyin.get_pinyin(label)
    new_label = ''.join(random.choices(rd, k=6))
    # new_label = '20191209'
    new_f = new_label + '_' + suffix + '.jpg'
    # print(new_f)

    old = os.path.join(root_path, f)
    new = os.path.join(root_path, new_f)
    os.rename(old, new)
    text = new_f + ',' + label.replace(' ', '')
    texts.append(text)
    copy_texts.append(text)

    if len(copy_texts) >= 1000:
        with open('./gen/cut_image_20191209_bakup.txt', 'a+', encoding='utf-8') as f:
            f.write('\n')
            f.write('\n'.join(copy_texts))
        copy_texts.clear()

    if 0 == i % 10000:
        print('{} / {}'.format(i, len_files))


with open('./gen/cut_image_20191209.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(texts))
