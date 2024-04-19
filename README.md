# Multi-Label Respiratory Disease Prediction      
Done by: Abigail Tan, Mandis Loh, Tan Kay Wee      

## Executive Summary     
While chest X-ray technology is widely used for disease detection, the interpretation of these images by radiologists presents challenges due to the complexity of thoracic conditions and the scarcity of qualified professionals. Leveraging deep learning models offers promising potential to enhance diagnostic efficiency and accuracy. However, addressing the complexity of multi-label disease classification remains a significant hurdle. We aim to investigate the feasibility of lightweight deep learning models for multi-label disease detection, comparing state-of-the-art architectures like ResNet, AlexNet, and DenseNet with custom models. The findings from this study aim to contribute valuable insights to the development of improved diagnostic tools, ultimately advancing patient care in lung disease detection.     

## Instructions to run the project      
## Setup
### Environment Setup

Ensure you have Python 3.9 or 3.10 installed. Create and activate your desired virtual environment:

### Conda
```
conda create -n yourenvname python=3.9 # or python=3.10
conda activate yourenvname
```

### Venv
```
python3.9 -m venv "yourenvname " # or python=3.10
source yourenvname/bin/activate
```

### Dependency Installation
Install Python dependencies:
```
# installs Python packages listed in requirements.txt
pip install -r requirements.txt
```

### CUDA
We highly recommend that you have CUDA installed on your system. You can follow the instructions provided in the PyTorch documentation [here](https://pytorch.org/get-started/locally/).

## Download Dataset
The dataset used in this project is the [NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345). To download the dataset, please run the ./src/download_data.py file. You will need about 42GB of space to download the entire dataset.         

## Run the Models
The notebooks used to run the models, as well as the definitions themselves are located in the /notebooks/ directory. The notebooks are split by the different models and are named as such: 
AlexNet.ipynb contains our implementation of the state-of-the-art AlexNet model
AlexNet_variation.ipynb contains our implementation of our custom AlexNet model

All the notebooks contain both training and evaluation sections and parameters for each model can be adjusted within the notebooks themselves.

The notebooks are organized into three different types:

Models Successfully Run: 
- AlexNet
- AlexNet_variation
- DenseNet_variation
- ResNet

Models Unable to Run: 
- DenseNet
- ResNet_variation
- VGG

Data Exploration: 
- data_processing
- exploration
   

## Loading Trained Models       
The models we trained can be found [here](https://drive.google.com/drive/folders/13Bj80AKrLALYRXMmSPRb4NUXGj9BHpry?usp=sharing). We have included the saved Pytorch model weights as well as the training and validation losses which we have presented in our report. To reproduce the results, please save the desired model to the ./models folder in the root directory and navigate to the corresponding model notebook in the ./notebooks directory. From there, run the testing code found in the notebook.       

