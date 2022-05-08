import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

def predict_age_gender_race(home_dir, image_files):
    # Load Models
    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load(os.path.join(home_dir, 'weights/res34_fair_align_multi_7_20190809.pt'), map_location=torch.device('cpu')))
    model_fair_7 = model_fair_7.cpu()
    model_fair_7.eval()
    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load(os.path.join(home_dir, 'weights/fairface_alldata_4race_20191111.pt'), map_location=torch.device('cpu')))
    model_fair_4 = model_fair_4.cpu()
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    gender_scores_fair = []
    age_scores_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []
    
    # Classify every image
    for index, image in enumerate(image_files):
        image = trans(image).view(1, 3, 224, 224).cpu()

        outputs = np.squeeze(model_fair_7(image).cpu().detach().numpy())
        # age
        age_outputs = outputs[9:18]
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))
        age_pred = np.argmax(age_score)
        age_scores_fair.append(age_score)
        age_preds_fair.append(age_pred)
        # gender
        gender_outputs = outputs[7:9]
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        gender_pred = np.argmax(gender_score)
        gender_scores_fair.append(gender_score)
        gender_preds_fair.append(gender_pred)
        # race
        outputs = np.squeeze(model_fair_4(image).cpu().detach().numpy())
        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)
        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([race_preds_fair_4,gender_preds_fair,age_preds_fair,race_scores_fair_4,gender_scores_fair,age_scores_fair]).T
    result.columns = ['race_preds_fair_4','gender_preds_fair','age_preds_fair','race_scores_fair_4','gender_scores_fair','age_scores_fair']
    
    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'
    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'   
    # race
    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    return [result[["age","gender","race4"]].iloc[i,:].tolist() for i in range(result.shape[0])]
