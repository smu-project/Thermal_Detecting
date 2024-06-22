from torchvision.datasets import ImageNet as ImageNet_


class ImageNet:
    def __init__(self,
                 root,
                 image_set='train',
                 download=False,
                 transforms=None):

        self.dataset = ImageNet_(root,
                                 split=image_set)

        self.transforms = transforms

    def __getitem__(self, index):
        img, target = self.dataset[index]

        input = img

        if self.transforms is not None:
            input, target = self.transforms(input, target)

        return input, target, img

    def __len__(self):
        return len(self.dataset)

