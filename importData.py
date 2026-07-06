
import pandas
from sklearn.model_selection import train_test_split
import os

def get_image(image_id,data_dir):
    path1= os.path.join(data_dir,"HAM10000_images_part_1",image_id+".jpg")
    if(os.path.exists(path1)):
        return path1
    path2= os.path.join(data_dir,"HAM10000_images_part_2",image_id+".jpg")
    return path2
def formdata():
    pathDir = "dataset"
    metadata = pandas.read_csv(pathDir + "/HAM10000_metadata.csv")
    metadata['path']=metadata['image_id'].apply(lambda x:get_image(x,pathDir))
    diagnosis_mapping = {
        'akiec': 0, 'bcc': 1, 'bkl': 2,
        'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }
    metadata['label'] = metadata['dx'].map(diagnosis_mapping)

    print("Всего изображений:", len(metadata))

    train_df, val_df = train_test_split(
        metadata,
        test_size=0.3,
        random_state=87,
        stratify=metadata['label']
    )

    print(f"Обучающая выборка: {len(train_df)}")
    print(f"Валидационная выборка: {len(val_df)}")
    return train_df, val_df
