# 中文词频统计
# 根据生成的地址等信息，用于姓名等字段的生成
# 最终目的是使得生成的数据中，高频汉字占大多数

import os
from collections import Counter

chars = ''

with open('./gen/cut_image_20191202.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
    # lines = data.split('\n')
    # print(len(data))
    for line in data:
        chars += line.strip('\n').split(',')[-1]
    print(len(chars))

fre = Counter(chars)
re = fre.most_common(3000)
type(re)

top_3000 = ''
for key, val in re:
    top_3000 += key

with open('./material/chinese_top_3000.txt', 'w', encoding='utf-8') as f:
    f.write(top_3000)

