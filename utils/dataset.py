import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import natsort


class VGGFace2_dataset(Dataset):
    """
        VGGFace2 Dataset Loader
    """
    def __init__(self, csv_path, img_path, train_augmentation=False, image_size=112, biometric_mode=None):
        """
            Configurations of VGGFace2 Dataset.

            :param csv_path: Path of csv file for soft-biometric attributes
            :param img_path: Path of face and periocular images
            :param train_augmentation: Usage of data augmentation approach, where default value is False
            :param image_size: Size of image, where default value is 112
            :param biometric_mode: Type of biometrics, where default value is None (options: multi, face, ocular/ocu)
        """
        self.image_size = image_size
        self.train_augmentation = train_augmentation
        self.biometric_mode = biometric_mode

        if train_augmentation:
            self.transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomResizedCrop(size=(self.image_size, self.image_size), scale=(0.8, 1.0),
                                             ratio=(0.75, 1.33)),
                transforms.RandomEqualize(),
                transforms.RandomGrayscale(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        data_object_path = csv_path.replace(".csv", ".npy")
        if os.path.isfile(data_object_path):
            with open(data_object_path, 'rb') as f:
                self.data_dict = np.load(f, allow_pickle=True)[()]
        else:
            self.data_info = pd.read_csv(csv_path, low_memory=False)
            self.data_dict = {}
            original_subject_ids = []
            self.num_subjects = 0
            for csv_idx in tqdm(range(len(self.data_info))):
                img = os.path.join(img_path, self.data_info.iloc[csv_idx, 0].replace("/0", "/face/0"))

                if int(str(self.data_info.iloc[csv_idx, 1]).replace("\\", "")) not in original_subject_ids:
                    original_subject_ids.append(int(str(self.data_info.iloc[csv_idx, 1]).replace("\\", "")))
                    self.num_subjects += 1
                self.data_dict[csv_idx] = {
                    "face_path": img,
                    "ocular_left_path": img.replace("face", "ocular_left"),
                    "ocular_right_path": img.replace("face", "ocular_right"),
                    "attribute": self.data_info.iloc[csv_idx, 2:].to_numpy().astype(np.int8),
                    "gt": original_subject_ids.index(int(str(self.data_info.iloc[csv_idx, 1]).replace("\\", "")))
                }
            print("Total number of subjects: {}".format(self.num_subjects))
            with open(data_object_path, 'wb') as f:
                np.save(f, self.data_dict)
            
    def __getitem__(self, idx):
        if self.train_augmentation:
            # Face images
            face_img = Image.open(self.data_dict[idx]['face_path']).convert('RGB')
            face_stacked_imgs = [self.transforms(face_img)]
            face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)
            # Ocular images
            ocular_l_img = Image.open(self.data_dict[idx]['ocular_left_path']).convert('RGB')
            ocular_r_img = Image.open(self.data_dict[idx]['ocular_right_path']).convert('RGB')

            ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
            ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

            return face_stacked_imgs, ocular_stacked_imgs, self.data_dict[idx]['attribute'], self.data_dict[idx]['gt']
        else:
            if self.biometric_mode == "multi":
                # Face images
                face_img = Image.open(self.data_dict[idx]['face_path']).convert('RGB')
                face_stacked_imgs = [self.transforms(face_img)]
                face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)
                # Ocular images
                ocular_l_img = Image.open(self.data_dict[idx]['ocular_left_path']).convert('RGB')
                ocular_r_img = Image.open(self.data_dict[idx]['ocular_right_path']).convert('RGB')

                ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
                ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

                return face_stacked_imgs, ocular_stacked_imgs, self.data_dict[idx]['gt']

            elif self.biometric_mode == "face":
                # Face images
                face_img = Image.open(self.data_dict[idx]['face_path']).convert('RGB')
                face_stacked_imgs = [self.transforms(face_img)]
                face_stacked_imgs = torch.stack(face_stacked_imgs, dim=0)

                return face_stacked_imgs, self.data_dict[idx]['gt']

            elif self.biometric_mode == "ocular" or self.biometric_mode == "ocu":
                # Ocular images
                ocular_l_img = Image.open(self.data_dict[idx]['ocular_left_path']).convert('RGB')
                ocular_r_img = Image.open(self.data_dict[idx]['ocular_right_path']).convert('RGB')

                ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
                ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

                return ocular_stacked_imgs, self.data_dict[idx]['gt']
            else:
                raise "biometric_mode is not selected !!!"

    def __len__(self):
        return len(self.data_dict)


class custom_multimodal_dataset(torch.utils.data.Dataset):
    """
        Custom Multimodal Dataset
    """
    def __init__(self, config, subdir="gallery/", train_augmentation=False, imagesize=112):
        """
            Configurations of Custom Multimodal Dataset.

            :param config: Configuration file
            :param subdir: Path directories for gallery or probe set, where default value is gallery/
            :param train_augmentation: Usage of data augmentation approach, where default value is False
            :param imagesize: Size of input image, where default value is 112
        """
        self.image_size = imagesize
        self.train_augmentation = train_augmentation

        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.data_path = natsort.natsorted(os.listdir(config.other_dataset_path_img + subdir))
        self.data_list, self.data_list_l, self.data_list_r = [], [], []
        self.labels = []
        label = 0

        for path in self.data_path:
            img_files = natsort.natsorted(os.listdir(os.path.join(config.other_dataset_path_img + subdir, path + '/face/')))
            for img in img_files:
                self.data_list.append(os.path.join(config.other_dataset_path_img + subdir, path + '/face/') + img)
                self.data_list_l.append(os.path.join(config.other_dataset_path_img + subdir, path + '/ocular_left/') + img)
                self.data_list_r.append(os.path.join(config.other_dataset_path_img + subdir, path + '/ocular_right/') + img)
                self.labels.append(label)
            label += 1

        self.num_samples = len(self.data_list_l)

    def __getitem__(self, idx):
        img = Image.open(self.data_list[idx]).convert('RGB')
        img = self.transforms(img).unsqueeze(dim=0)

        ocular_l_img = Image.open(self.data_list_l[idx]).convert('RGB')
        ocular_r_img = Image.open(self.data_list_r[idx]).convert('RGB')
        subjects = self.labels[idx]

        ocular_stacked_imgs = [self.transforms(ocular_l_img), self.transforms(ocular_r_img)]
        ocular_stacked_imgs = torch.stack(ocular_stacked_imgs, dim=0)

        return img, ocular_stacked_imgs, subjects

    def __len__(self):
        return self.num_samples
