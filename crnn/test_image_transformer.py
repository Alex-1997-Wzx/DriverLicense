
from PIL import Image
# import dataset

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        # self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        return img

img_path = './data/244.jpg'
transformer = resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)

image.save('test.jpg')