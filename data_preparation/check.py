import glob
from PIL import Image

imagepath = 'D:/ЧелГУ/Практика/Фото/'

imgs_names = glob.glob(imagepath+'\\*.jpg')

for imgname in imgs_names:
    img = Image.open(imgname)
    if img is None:
        print(imgname)