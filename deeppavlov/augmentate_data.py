from pathlib import Path

import pandas as pd

from deeppavlov import build_model, configs


if __name__ == '__main__':
    aug_model = build_model(configs.augmentation.en_thesaurus_aug)
    for name_source_path in ['en_source_imdb', 'en_source_twitter_airlines', 'en_source_toxic']:
        source_data = pd.read_csv(Path("/home/azat/.deeppavlov/downloads/experiments") / name_source_path)
        print(f"input file: {Path('/home/azat/.deeppavlov/downloads/experiments' / name_source_path)}")
        print(f"source len: {len(source_data)}")
        augmented_text = aug_model(source_data['text'])
        aug_data = pd.DataFrame(data=zip(source_data['label'], augmented_text), columns=['label', 'text'])
        plus = pd.concat([aug_data, source_data])
        print(f"new len: {len(plus)}")
        plus.to_csv(Path("/home/azat/.deeppavlov/downloads/experiments") / name_source_path.replace('source', 'aug'))
        print(f"output file: {Path('/home/azat/.deeppavlov/downloads/experiments') / name_source_path.replace('source', 'aug')}")
        print()