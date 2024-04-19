# Multi-Label Respiratory Disease Prediction      
Done by: Abigail Tan, Mandis Loh, Tan Kay Wee      

## Executive Summary     
While chest X-ray technology is widely used for disease detection, the interpretation of these images by radiologists presents challenges due to the complexity of thoracic conditions and the scarcity of qualified professionals. Leveraging deep learning models offers promising potential to enhance diagnostic efficiency and accuracy. However, addressing the complexity of multi-label disease classification remains a significant hurdle. We aim to investigate the feasibility of lightweight deep learning models for multi-label disease detection, comparing state-of-the-art architectures like ResNet, AlexNet, and DenseNet with custom models. The findings from this study aim to contribute valuable insights to the development of improved diagnostic tools, ultimately advancing patient care in lung disease detection.     

## Instructions to run the project      
### Setting up the project        
Install the necessary packages    

```
  pip install -r requirements.txt       
```

### Downloading the dataset
The dataset used in this project is the [NIH Chest X-ray Dataset of 14 Common Thorax Disease Categories](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/36938765345). To download the dataset, please run the ./src/download_data.py file. 

### Run the Models
The notebooks used to run the models, as well as the model definitions themselves can be found within ./notebook folder. The notebooks are split by the different models, and are named as such. For example, AlexNet.ipynb will contain our implementation of the AlexNet model, as well as the training and evaluation for the model. Parameters for each model can be adjusted within the notebooks themselves.          

Within the ./notebook folder, we have also included a folder of models which we failed to run for this project.       

### Loading Trained Models       
The models we trained can be found [here](https://drive.google.com/drive/folders/13Bj80AKrLALYRXMmSPRb4NUXGj9BHpry?usp=sharing). We have included the saved Pytorch model weights as well as the training and validation losses which we have presented in our report. To reproduce the results, please save the desired model to the ./models folder in the root directory and navigate to the corresponding model notebook in the ./notebooks directory. From there, run the testing code found in the notebook.       

