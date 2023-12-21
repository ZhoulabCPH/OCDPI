import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd
import torch, torchvision
import torch.nn as nn
import feather
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from ctran import ctranspath

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose(
    [
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
)
class roi_dataset(Dataset):
    def __init__(self, img_csv,
                 ):
        super().__init__()
        self.transform = trnsfrms_val

        self.images_lst = img_csv

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        img_name = self.images_lst.iloc[idx, 0]
        path = self.images_lst.iloc[idx,1]

        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return img_name, image


def get_CTransPath_features(slides_path, save_path):
    """Extract the representation vector of patches.
    """
    model = ctranspath()
    model.head = nn.Identity()
    td = torch.load(r'../checkpoints/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    model.eval()
    model.cuda()
    count = 0
    for slide in os.listdir(slides_path):
        count = count + 1
        print(f'{count}-th {slide}')
        if f'{slide}.feather' in os.listdir(save_path):
            continue
        patches_name = list(os.listdir(f'{slides_path}/{slide}'))
        patches_path = [f'{slides_path}/{slide}/{patch_name}' for patch_name in patches_name]
        slide_patches = pd.DataFrame()
        slide_patches['patches_name'] = patches_name
        slide_patches['patches_path'] = patches_path
        test_datat = roi_dataset(slide_patches)
        database_loader = torch.utils.data.DataLoader(test_datat, batch_size=32, shuffle=False)
        patches_name = []
        patches_features = []
        with torch.no_grad():
            for batch_patches_name, batch_patches_features in database_loader:
                features = model(batch_patches_features.cuda())
                features = features.cpu().numpy()
                patches_name = patches_name + list(batch_patches_name)
                patches_features = patches_features + list(features)
            slide_features = pd.DataFrame(data = patches_features)
            slide_features.index = patches_name
            feather.write_dataframe(slide_features, f'{save_path}/{slide}.feather')

