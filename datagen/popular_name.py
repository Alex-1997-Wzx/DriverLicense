''' 生成受欢迎的名字(词频排序) '''
import os


root_dir = './material/names'

files = os.listdir(root_dir)

firstnames = []
secondnames = []

for f in files:
    fn = os.path.join(root_dir, f)
    with open(fn, 'r', encoding='utf-8') as hd:
        data = hd.readlines()
        for line in data:
            l = line.strip('\n')
            firstnames.append(l[0])
            secondnames.append(l[1:])

len(firstnames)
len(secondnames)

from collections import Counter
f_c = Counter(firstnames)
f_200 = f_c.most_common(200)
s_c = Counter(secondnames)
s_5000 = s_c.most_common(5000)

with open('firstname_top_200.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([x[0] for x in f_200]))

with open('secondname_top_5000.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join([x[0] for x in s_5000]))