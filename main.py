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

st.title('PyTorch Paddy Disease Classifier')
img=st.file_uploader("Upload an Image to predict")
if img is not None and st.button("Predict"):
    img=Image.open(img).convert('RGB')
    st.image(img,width=200)
    img=transform(img)
    img=img.unsqueeze(0)
    img=img.to(device)
    pred=model(img)
    name = torch.argmax(pred, axis=1).item()
    lbl = label_encoder.inverse_transform([name])[0]
    if lbl == 'normal':
        st.success('This Sample has no disease')
    else:
        st.success(f"Predicted disease is {lbl}")
 else:
    st.error('Please upload an image')