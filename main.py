import os
from PIL import Image
from embedding import FeatureExtractor
from similarity import Cosine_similarity

import warnings
warnings.filterwarnings(action="ignore")

if __name__ == "__main__":
    
    root_dir = "./data/images/images"
    files =  os.path.join(root_dir)
    File_names = os.listdir(files)

    fe = FeatureExtractor()

    try:
        image1_path = os.path.join(root_dir, 'abomasnow.png')
        image2_path = os.path.join(root_dir, 'abomasnow.png')

        image1 = fe.get_extract(img=Image.open(image1_path))
        image2 = fe.get_extract(img=Image.open(image2_path))

        cosine_similarity = Cosine_similarity._cos_sin(image1, image2)
        print(f"두 이미지의 유사도:", round(cosine_similarity, 2))
    except Exception as e:
        print('예외가 발생했습니다.', e)