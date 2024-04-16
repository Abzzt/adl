import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import os
import numpy as np
from itertools import chain
import pandas as pd
from torchvision import transforms


image_size = 224
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

bb_pd = pd.read_csv('data/BBox_List_2017.csv', delimiter=',')
data_entry = pd.read_csv('data/Data_Entry_2017_v2020.csv', delimiter=',')

labels = np.unique(list(chain(*data_entry['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]
for label in labels:
    if len(label) > 1:
        data_entry[label] = data_entry['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
        
labels_map = {i: label for i, label in enumerate(labels)}
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

# Define transforms
train_transform = transforms.Compose([
   transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

transform_data = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))])

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.dataset_length = len(self.dataframe)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        try:
            img_path = self.dataframe.iloc[idx]['path']
            image = Image.open(img_path).convert('RGB')

            # Convert string representation of list to actual list
            # Convert the list of labels to a multi-hot encoded tensor
            num_classes = len(labels_map)
            multi_hot_labels = [1 if label in labels else 0 for label in labels_map.values()]
            multi_hot_labels = torch.tensor(multi_hot_labels, dtype=torch.float32)

            if self.transform:
                image = self.transform(image)

            # return image, multi_hot_labels
            
            return {'image': image, 'labels': multi_hot_labels}
        except Exception as e:
            print(f"Error loading data at index {idx}: {e}")
            return None, None 


# Create datasets
train_dataset = CustomDataset(train_df, transform=transform_data)
valid_dataset = CustomDataset(valid_df, transform=transform_data)

img, lab = train_dataset[0] # torch, list types

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

print(type(train_loader))
print(len(train_loader))


for i_batch, sample_batched in enumerate(train_loader):
    print(i_batch, sample_batched['image'],
          sample_batched['labels'])

for i_batch, sample_batched in enumerate(valid_loader):
    print(i_batch, sample_batched['image'],
          sample_batched['labels'])
