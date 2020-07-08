import os

label_dir = './gen/labels'
image_dir = './gen/original'

images = []
for f in os.listdir(image_dir):
    images.append(f[:-4])

labels_rm = []
for f in os.listdir(label_dir):
    if f[:-4] not in images:
        labels_rm.append(os.path.join(label_dir, f))

print('len of to rm: ', len(labels_rm))

for f in labels_rm:
    os.remove(f)