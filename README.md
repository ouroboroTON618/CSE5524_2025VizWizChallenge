# CSE5524_2025VizWizChallenge

## Structure:
- data/Test folder: Containing all .jpg images
- data/test_grounding.json: image id - question pairs
- trained_model/best_advanced_model.pth: trained model
- model.py: advanced model architecture
- predict.ipynb: jupyternotebook to run model with few examples

## How to run
1. Run command: pip install -r requirements.txt
2. Open the jupyter notebook called predict.ipynb
3. Select your preferred (python) kernel to use jupyternotebook
4. Run all the cells

## Output
- It should generate 5 images with its question, and a mask over the object that answers the question
- The mask on top is a result of the trained advanced model