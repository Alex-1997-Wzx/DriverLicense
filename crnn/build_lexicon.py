import os


root_dir = r'D:\xiashu\OCR\jutze\data\cut_image'
lexicon_fname = 'lexicon.txt'

filenames = os.listdir(root_dir)
names = [os.path.splitext(f)[0] for f in filenames]
lexicons = [name.split('_')[-1] for name in names]

lexicons = list(set(lexicons))

with open(lexicon_fname, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lexicons))