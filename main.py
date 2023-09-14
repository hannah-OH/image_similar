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

    print('>>>>> "사전에 훈련된 VVG16 모델을 이용하여 포켓몬 유사도 구하기 프로그램입니다 <<<<<')
    print('원하시는 두 포켓몬을 입력하여, 유사도를 구할 수 있습니다.')


    pokemon_1 = input('첫번째 포켓몬 이름을 입력하세요: ')
    pokemon_2 = input('두번째 포켓몬 이름을 입력하세요: ')

    pokemon_1 = pokemon_1.lower()
    pokemon_2 = pokemon_2.lower()

    pokemon_name_1 = pokemon_1 + '.png'
    pokemon_name_2 = pokemon_2 + '.png'

    image1_path = os.path.join(root_dir, pokemon_name_1)
    image2_path = os.path.join(root_dir, pokemon_name_2)

    if not os.path.exists(image1_path):
        pokemon_name_1 = pokemon_1 + '.jpg'
        image1_path = os.path.join(root_dir, pokemon_name_1)

    if not os.path.exists(image2_path):
        pokemon_name_2 = pokemon_2 + '.jpg'
        image2_path = os.path.join(root_dir, pokemon_name_2)

    try:
        image1 = fe.get_extract(img=Image.open(image1_path))
        image2 = fe.get_extract(img=Image.open(image2_path))

        cosine_similarity = Cosine_similarity._cos_sin(image1, image2)

        print(f"두 이미지의 유사도:", round(cosine_similarity, 2))

    except FileNotFoundError:
        print('포켓몬 이름이 잘못들어갔습니다. 다시 확인해주세요.')

    except Exception as e:
        print('예외가 발생했습니다.', e)

