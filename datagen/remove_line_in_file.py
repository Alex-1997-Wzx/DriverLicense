import os

root = './data/labeled/txt'
for f in os.listdir(root):
    fn = os.path.join(root, f)
    with open(fn, "r+", encoding='utf-8') as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if 'Driving' not in str(i):
                f.write(i)
            else:
                print(f)
                print(i)
        f.truncate()