import torch
import pandas as pd
from dataset import OV_, Transform_
from model import BarlowTwins_
from torch.utils.data import DataLoader
from model import BarlowTwins as BarlowTwins


# Get pretrained ResNet feature
def get_features(model_dir, store_dir,label_dir, save_dir):
    print('Features extract!')
    model = BarlowTwins_()
    ckpt = torch.load(model_dir, map_location='cuda:1')
    model.load_state_dict(ckpt['model'])
    model.to(torch.device('cuda:1'))
    model.eval()
    data = OV_(store_dir=store_dir, label_dir=label_dir, transform=Transform_())
    data_loader = DataLoader(data, 128, shuffle=False, num_workers=8, drop_last=False)
    data = pd.DataFrame()
    for i, (features, label) in enumerate(data_loader):
        features = features.to(torch.device('cuda:1'))
        latent = model(features)
        latent_list = latent.detach().cpu().numpy().tolist()
        label = list(label)
        if len(label) != len(latent_list):
            print('Warning!')
        label_latent = [[label[i]] + latent_list[i] for i in range(len(label))]
        data_ = pd.DataFrame(data=label_latent)
        data = pd.concat([data, data_])
    data.to_csv(save_dir)

    print('Done!')


def get_model(args):
    model_ = BarlowTwins(args)
    pretrained_state_dict = torch.load(
        args.pretrained_dir,
        map_location='cuda:0')
    model_state_dict = model_.state_dict()
    pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
    model_state_dict.update(pretrained_state_dict)
    model_.load_state_dict(model_state_dict)
    return model_