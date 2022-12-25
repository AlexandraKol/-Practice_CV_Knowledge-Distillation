import Augmentor
import os
import splitfolders

dir_name = "D:/ЧелГУ/Практика/Фото/"
folders = os.listdir(dir_name)

for f in range(len(folders)):
     folder_path = os.path.join(dir_name, folders[f])
     p = Augmentor.Pipeline(source_directory = folder_path, output_directory= folder_path)
     p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=8)
     p.random_brightness(probability=0.9, min_factor=0.3, max_factor=1.2)
     p.random_color(probability=0.5, min_factor=0.5, max_factor=0.7)
     p.random_contrast(probability=0.5, min_factor=0.5, max_factor=0.7)
     p.sample(20-len(os.listdir(folder_path)))
     print(len(os.listdir(folder_path)))

output_folder = "D:/ЧелГУ/Практика/data"
splitfolders.ratio(dir_name, output_folder, ratio = (0.7, 0.2, 0.1), seed=13, group_prefix=None)