from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
from torchvision import transforms
from src.models.ensemble import CombinedModel
import io

app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CombinedModel(n_classes=3)
model.load_state_dict(torch.load('best.pth', map_location=device))
model.to(device).eval()

tfm = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.softmax(model(x), dim=1)
        cls = preds.argmax(1).item()
    return {'class_id': int(cls), 'confidence': float(preds[0][cls])}
