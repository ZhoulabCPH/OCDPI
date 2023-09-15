import tables
import os
import numpy as np
import pandas as pd
from PIL import Image


def make_hdf5(slides_dir, hdf5_dir):
    count = 0
    count_slide = 0
    store = tables.open_file(hdf5_dir, mode='w')
    img_dtype = tables.UInt8Atom()
    data_shape = (0, 224, 224, 3)
    storage = store.create_earray(store.root, atom=img_dtype, name='patches', shape=data_shape)
    patches_name =[]
    for slide_name in os.listdir(slides_dir):
        count_slide = count_slide+1
        if count_slide % 100 == 0:
            print(count_slide)

        path = slides_dir+slide_name+'/'
        for img in sorted(os.listdir(path)):
            img_name = img
            img = np.array(Image.open(path+img))
            if img.shape == (224, 224, 3):
                if count % 10000 == 0:
                    print(count)
                count = count+1
                storage.append(img[None])
                patches_name.append(img_name)

    storage.close()
    patient_5_patches = pd.DataFrame()
    patient_5_patches['patches'] = patches_name
    patient_5_patches.to_csv('./patches.csv')
