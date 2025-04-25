# CSE5524_2025VizWizChallenge

## Structure:
- data/test: Containing all .jpg images
- data/test_grounding.json: image id - question pairs
- trained_model/best_advanced_model.pth: trained model
- model.py: advanced model architecture
- predict.ipynb: jupyternotebook to run model with few examples

## Environment
- Windows (commands based on)
- Visual Studio Code

## Before running:
- Click in download zip, once downloaded unzip it and place it inside your VSCode
  
![image](https://github.com/user-attachments/assets/6d4d8c01-a1be-48e4-9441-3e4ebd91f8e5)

 - Download the trained model .pth file at: https://drive.google.com/file/d/1oaskN2klTiIk9tRfMROS4ZH7yaoqqb2d/view?usp=sharing
- It will take around 6 minutes to download (couldnt commit to git because is large). If download not working or file failed, please contact me. You can also see directly the results in this github repo that I obtained, by clicking in the notebook predict.ipynb
- Place it inside directory called "trained_model"


## How to run
1. Run command: pip install -r requirements.txt
2. Open the jupyter notebook called predict.ipynb
3. Select your preferred (python) kernel to use jupyternotebook
4. Run all the cells

## Output
- It should generate 5 images with its question, and a mask over the object that answers the question

  ![image](https://github.com/user-attachments/assets/d55a36de-1d76-4d0d-ac2c-87100582ebff)

