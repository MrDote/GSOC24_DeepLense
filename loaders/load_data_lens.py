from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


import numpy as np

import os



class _MinMaxNormalizeImage:
    def __call__(self, img: torch.Tensor):
        min_val = img.min()
        max_val = img.max()
        normalized_tensor = (img - min_val) / (max_val - min_val)
        return normalized_tensor
        



class _LensData(Dataset):
    def __init__(self, train=True, transform=None, class_samples = 10000):

        self.transform = transform

        # Initialize lists for each folder
        FOLDERS = ['no', 'sphere', 'vort']

        if train:
            prefix = "data/train"
        else:
            prefix = "data/val"


        data = []
        labels = []

        for (i, folder) in enumerate(FOLDERS):

            files = os.listdir(os.path.join(prefix, folder))

            for (k, file) in enumerate(files):
                if class_samples == k:
                    break
                im = np.load(os.path.join(prefix, folder, file))
                data.append(im)
                labels.append(i)

        
        data = np.array(data, dtype=np.float32).transpose((0, 2, 3, 1))
        labels = np.array(labels, dtype=np.int32)



        #* For testing purposes only
        # for (i, label) in enumerate(labels):
        #     if label == 0:
        #         data[i, 60:63, :] = 1

        #     if label == 1:
        #         data[i, 60:63, :] = 0


        # np.random.shuffle(labels)

        # unique_elements, counts = np.unique(labels, return_counts=True)
        # occurrences = dict(zip(unique_elements, counts))
        # print(occurrences)
        
        
        # data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.1, random_state=42, shuffle=False)
                
        indices = np.arange(len(data))
        # np.random.shuffle(indices)

        self.data = data[indices]
        self.labels = labels[indices]



    #len(dataset) returns the size of the dataset
    #The __len__ function returns the number of samples in our dataset
    def __len__(self):
        return len(self.labels)


    #This will return a given image and a corrosponding index for the image
    #__getitem__ to support the indexing such that dataset[i] can be used to get ith sample.
    def __getitem__(self, i):
        # img = io.imread(self.data[index])
        # print(self.data[index].shape)
        # print(np.squeeze(self.data[index], axis=2).shape)

        img = self.data[i]

        labels = self.labels[i]

        if self.transform is not None:
            img = self.transform(img)

        return img, labels





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    class EdgeDetectionTransform:
        def __init__(self, kernel_size=3):
            self.kernel_size = kernel_size

            # Define edge detection kernel
            # self.kernel = torch.tensor([[-1, -1, -1],
            #                             [-1,  8, -1],
            #                             [-1, -1, -1]], dtype=torch.float32)
            
            self.kernel = torch.tensor([[0, -1, 0],
                                        [-1,  4, -1],
                                        [0, -1, 0]], dtype=torch.float32)

        def __call__(self, img: torch.Tensor):
            # Apply 2D convolution with edge detection kernel
            edge_tensor = torch.nn.functional.conv2d(img.unsqueeze(0), 
                                                    self.kernel.unsqueeze(0).unsqueeze(0), 
                                                    padding=self.kernel_size // 2)
            
            # Normalize the output tensor
            edge_tensor = torch.clamp(edge_tensor, 0, 1)  # Clip values to [0, 1]
            
            return edge_tensor.squeeze(0)
    


    class AdjustBrightness:
        def __init__(self, factor) -> None:
            self.factor = factor

        def __call__(self, img: torch.Tensor):
            adjusted_tensor = 1 / (1 + torch.exp(-self.factor * (img - 0.5)))
            return adjusted_tensor



    samples = 3
    dataset = _LensData(
        train=True,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(100),
            transforms.Resize(100),
            transforms.ToTensor(),
            EdgeDetectionTransform(),
            _MinMaxNormalizeImage(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            # AdjustBrightness(2)
        ]),
        class_samples=samples
    )

    IMGS_IN_ROW = 3

    # print(len(dataset))

    fig, axes = plt.subplots(int(len(dataset)/IMGS_IN_ROW), IMGS_IN_ROW, sharex='all', sharey='all', figsize=(20,15))
    plt.axis('off')

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        print(f"{i} --> ", dataset[i][1])
        print(torch.max(dataset[i][0][0, :, :]))
        print(torch.min(dataset[i][0][0, :, :]))
        ax.imshow(dataset[i][0][0, :, :])


    plt.tight_layout()
    plt.show()




class Lens:

    def __init__(self, batch_size=1, num_workers=1, crop_size=150, img_size=150, rotation_degrees=0, translate=(0.0, 0.0), scale=(1.0, 1.0), *, class_samples):

        self.batch_size = batch_size

        self.num_workers = num_workers

        self.crop_size = crop_size
        self.img_size = img_size
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale

        self.class_samples = class_samples


    def __call__(self):

        train_loader = DataLoader(
            _LensData(
                train=True,
                
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(self.crop_size),
                    transforms.Resize(self.img_size),
                    transforms.RandomAffine(
                        degrees=self.rotation, 
                        translate=self.translate,
                        scale=self.scale
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    _MinMaxNormalizeImage()
                ]),
                class_samples=self.class_samples
            ),
            
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


        test_loader = DataLoader(
            _LensData(
                train=False,
                
                transform=transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.CenterCrop(self.crop_size),
                    transforms.Resize(self.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                    _MinMaxNormalizeImage()
                ]),
                class_samples=self.class_samples/5
            ),
            
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, test_loader, self.img_size