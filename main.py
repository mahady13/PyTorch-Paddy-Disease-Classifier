from model_architecture import MyModel
import torch
import torchvision.transforms as transforms
import joblib
import streamlit as st
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

device='cuda' if torch.cuda.is_available() else 'cpu'
def load_models():
    model=MyModel()
    checkpoint=torch.load("best_model.pth",map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    label_encoder=joblib.load("label_encoder (2).pkl")

    return model,label_encoder

model,label_encoder=load_models()
transform=transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32)
])