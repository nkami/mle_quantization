import torchvision
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch
from src.quantized_ops import quantize_model
from tqdm import tqdm
import argparse


possible_models = {'VGG16': models.vgg16(pretrained=True),
                   'VGG16_BN': models.vgg16_bn(pretrained=True),
                   'ResNet18': models.resnet18(pretrained=True),
                   'ResNet50': models.resnet50(pretrained=True),
                   'ResNet101': models.resnet101(pretrained=True),
                   'Inception_V3': models.inception_v3(pretrained=True)}


class ImageNetValidationDataset(Dataset):
    def __init__(self, images_path: str, labels_path: str, pytorch_map: bool = True, ds_transforms: Compose = None):
        self.images_path = images_path
        with open(labels_path + 'ILSVRC2012_validation_ground_truth.txt', 'r') as f:
            self.labels = f.readlines()
            self.labels = [int(label.strip()) for label in self.labels]
        if pytorch_map:
            with open(labels_path + 'ILSVRC2012_validation_ground_truth_labels.txt', 'r') as f:
                labels_to_keys = f.readlines()
                labels_to_keys = {id + 1: line.strip().split(' ')[0] for id, line in enumerate(labels_to_keys)}
            with open(labels_path + 'pytorch_pretrained_id_to_ground_truth.txt', 'r') as f:
                key_to_pytorch_id = f.readlines()
                key_to_pytorch_id = {line.strip(): id + 1 for id, line in enumerate(key_to_pytorch_id)}
            self.labels = [key_to_pytorch_id[labels_to_keys[label]] for label in self.labels]
        self.ds_transforms = ds_transforms

    def __len__(self):
        return 50000

    def __getitem__(self, item):
        img_name = self.images_path + 'ILSVRC2012_val_' + str(item + 1).zfill(8) + '.JPEG'
        x = Image.open(img_name).convert('RGB')
        if self.ds_transforms is not None:
            x = self.ds_transforms(x)
        y = torch.tensor([self.labels[item]])
        return x, y


def test_model(model: nn.Module, data_loader: DataLoader, device: torch.device, quantize: bool = True,
               w_bits: int = None, a_bits: int = None) -> float:
    model = model.to(device)
    if quantize:
        quantize_model(model, w_bits, a_bits)
    model.eval()
    accuracies = []
    early = 0
    for x, y in tqdm(data_loader):
        early += 1
        x, y = x.to(device), y.to(device)
        probs = model(x)
        preds = torch.argmax(probs, dim=1)
        accuracies.append(torch.sum(preds + 1 == y).item() / y.shape[0])
        if early >= (50 / 1):
            break
    return sum(accuracies) / len(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='possible names: VGG16, VGG16_BN, ResNet18, ResNet50, ResNet101, Inception_V3')
    parser.add_argument('--quantize', dest='quantize', action='store_true', default=False, help='quantize model')
    parser.add_argument('--w_bits', type=int, help='number of bits for weights')
    parser.add_argument('--a_bits', type=int, help='number of bits for activations')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using: ' + device.type)
    model_name = args.model
    weight_bits = args.w_bits
    act_bits = args.a_bits
    quantize = args.quantize
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = 256 if model_name != 'Inception_V3' else 299
    crop_size = 224 if model_name != 'Inception_V3' else 299
    ds_transforms = [
        transforms.Resize(resize),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ]
    dataset_transforms = transforms.Compose(ds_transforms)
    imagenet_ds = ImageNetValidationDataset('./imagenet/ILSVRC2012/validation_data/ILSVRC2012_imgs/',
                                            './imagenet/ILSVRC2012/validation_data/ILSVRC2012_labels/',
                                            ds_transforms=dataset_transforms)
    data_loader = DataLoader(imagenet_ds, batch_size=1, shuffle=False)
    acc = test_model(possible_models[model_name], data_loader, device, quantize=quantize,
                     w_bits=weight_bits, a_bits=act_bits)
    print(model_name + ' got an accuracy of: ' + str(acc))
