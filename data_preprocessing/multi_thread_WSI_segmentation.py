import os
import concurrent.futures
from histolab.slide import Slide
from histolab.tiler import GridTiler

def tiling_WSI_OV_224(slide_path, output_path):
    """We modified histolab to implement WSI segmentation.
       """
    grid_tiles_extractor = GridTiler(
        tile_size=(224, 224),
        level=0,
        check_tissue=True,
        tissue_percent=50,
        pixel_overlap=0,
        prefix="",
        suffix=".png"
    )

    slide = Slide(slide_path, output_path)
    grid_tiles_extractor.extract(slide)


def process_slide(slide):
    slides_path = f'../datasets/WSIs/TCGA_OV/{slide}/'
    output_path = f'../datasets/patches/TCGA_OV/{slide}/'

    if not os.path.exists(output_path) or os.listdir(output_path) == []:
        os.makedirs(output_path)
        print(f'Tiling {slide}.')
        tiling_WSI_OV_224(slides_path, output_path)
        print(f'{slide} finish!')


if __name__ == '__main__':
    slides = os.listdir('../datasets/WSIs/TCGA_OV/')
    num_threads = 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_slide, slides)



