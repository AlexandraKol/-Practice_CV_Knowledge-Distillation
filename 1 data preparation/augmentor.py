import Augmentor

p = Augmentor.Pipeline("D:/ЧелГУ/Практика/Фото/Фото ценника")
p.random_distortion(probability=0.9, grid_width=4, grid_height=4, magnitude=8)
p.random_brightness(probability=0.9, min_factor=0.3, max_factor=1.2)
p.random_color(probability=0.5, min_factor=0.5, max_factor=0.7)
p.random_contrast(probability=0.5, min_factor=0.5, max_factor=0.7)
p.flip_left_right(probability=0.3)

p.sample(1970)
