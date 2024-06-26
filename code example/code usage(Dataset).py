import sys
import os

lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.'))
sys.path.append(lib_dir)

from torch.utils.data import DataLoader
from utils.dataset import VGGFace2_dataset


dataset = VGGFace2_dataset(csv_path="./data/MAAD_Face_filtered_train.csv",
                           img_path='./data/VGGFace2/train/',
                           train_augmentation=True, image_size=112)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

data_iter = iter(data_loader)
for i in range(len(data_iter)):
    face_img, ocular, attribute, subject = next(data_iter)
    print(face_img.shape)
    print(ocular.shape)
    print(attribute.shape)
    print(subject)
