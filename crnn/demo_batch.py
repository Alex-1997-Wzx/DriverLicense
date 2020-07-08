from __future__ import division
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


# model_path = './data/crnn.pth'
# img_path = './data/demo.png'
model_path = './expr/netCRNN_24_120.pth'
img_path = './data/cut_image_val'

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()

model = torch.nn.DataParallel(model)
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))


import os
total = 0
acc = 0
for f in os.listdir(img_path):
    label = f[:-4].split('_')[-1]
    label = label.lower()

    fn = os.path.join(img_path, f)

    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(fn).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('')
    print('truth: ', label)
    # print len(label)
    # print(sim_pred)
    # print len(sim_pred)
    print('%-20s => %-20s' % (raw_pred, sim_pred))

    total += 1
    if label == sim_pred:
        acc += 1


print('\n')
print(acc)
print(total)
print('acc = ', acc/total) 
