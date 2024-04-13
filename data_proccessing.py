from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np
from itertools import chain
import pandas as pd

image_size = 224
batch_size = 32

bb_pd = pd.read_csv('data/BBox_List_2017.csv', delimiter=',')
data_entry = pd.read_csv('data/Data_Entry_2017_v2020.csv', delimiter=',')

labels = np.unique(list(chain(*data_entry['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]
for label in labels:
    if len(label) > 1:
        data_entry[label] = data_entry['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
        
# looks for files with names matching
image_directory = 'data/images/images'
data_image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(image_directory, '*.png'))}
data_entry['path'] = data_entry['Image Index'].map(data_image_paths.get)
df_new = pd.DataFrame(columns=data_entry.columns)

for l in labels:
    df_new = pd.concat([df_new, data_entry[data_entry[l]==1][:500]], ignore_index=True)

df_new = df_new.drop_duplicates()

train_df, valid_df = train_test_split(df_new, test_size=0.20, random_state=2020, stratify=df_new['Finding Labels'].map(lambda x: x[:4]))

train_df.loc[:, 'labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df.loc[:, 'labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
core_idg = ImageDataGenerator(rescale=1 / 255,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.2,
                                  width_shift_range=0.2,
                                  rotation_range=10,
                                  shear_range=0.1,
                                  fill_mode='nearest',
                                  zoom_range=0.15)

core_idg2 = ImageDataGenerator(rescale=1 / 255)

train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

valid_gen = core_idg2.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=len(valid_df),
                                             classes=labels,
                                             target_size=(image_size, image_size))

X_val, y_val = next(valid_gen)
print(X_val, y_val)