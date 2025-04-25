# CSE5524_2025VizWizChallenge

## Structure:
- data/Test folder: Containing all .jpg images
- data/test_grounding.json: image id - question pairs
- trained_model/best_advanced_model.pth: trained model
- model.py: advanced model architecture
- predict.ipynb: jupyternotebook to run model with few examples

## Before running:
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
- The mask on top is a result of the trained advanced model
