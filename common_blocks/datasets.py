import skimage.io
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.df = df
        self.labels = df[config["target_col"]]
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["image_filename"].values[idx]
        file_path = "{}/{}".format(self.config["images_path"], file_name)
        image = skimage.io.MultiImage(file_path)[-1]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.labels[idx]
        class_to_id = {"correct": 1, "incorrect": 0}
        return image, float(class_to_id[label])


class TestDataset(Dataset):

    def __init__(self, df, config, transform=None):
        self.df = df
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["image_filename"].values[idx]
        file_path = "{}/{}".format(self.config["images_path"], file_name)
        image = skimage.io.MultiImage(file_path)[-1]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return file_path, image
