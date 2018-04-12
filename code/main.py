import torch
import torchvision as tvs

if __name__ == "__main__":
    with torch.cuda.device(1):
        # Normalize takes in arguments mean, std
        cifar_transform = tvs.transforms.Compose(
            [tvs.transforms.ToTensor(),
             tvs.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        cifar_train_dataset = tvs.datasets.CIFAR10(root='../data/', train=True,
                                                   download=True, transform=cifar_transform)
        cifar_train_loader = torch.utils.data.DataLoader(cifar_train_dataset,
                                                         batch_size=4, shuffle=True,
                                                         num_workers=2)

        cifar_test_dataset = tvs.datasets.CIFAR10(root='../data/', train=False,
                                                  download=True, transform=cifar_transform)
        cifar_test_loader = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=4,
                                                        shuffle=False, num_workers=2)
