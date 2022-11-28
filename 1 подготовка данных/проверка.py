import glob
from PIL import Image

imagepath = '/content/drive/MyDrive/data/train/Фото ценника/'

imgs_names = glob.glob(imagepath+'\\*.jpg')

for imgname in imgs_names:
    img = Image.open(imgname)
    if img is None:
        print(imgname)