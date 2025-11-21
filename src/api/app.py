from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
from src.models.ensemble import CombinedModel
from torchvision import transforms

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000", "http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
MODEL_PATH = "best.pth"
CLASS_NAMES = ["Early Blight","Healthy" ,"Late Blight", ]
THRESHOLD = 0.9  # confidence cutoff for "Unknown"

# --- LOAD MODEL ONCE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel(n_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# --- TRANSFORM ---
transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    # Read and preprocess image
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = torch.softmax(model(x), dim=1)
        cls = preds.argmax(1).item()
        confidence = preds[0][cls].item()

    # --- Apply threshold for "Unknown" ---
    if confidence < THRESHOLD:
        label = "Unknown"
    else:
        label = CLASS_NAMES[cls]

    # --- Response format matches frontend ---
    response = {
        "class_name": label,  # 👈 changed from "prediction" to "class_name"
        "confidence": round(confidence, 4),  # in 0–1 range
        "threshold": THRESHOLD,
        "class_probabilities": {
            CLASS_NAMES[i]: round(float(preds[0][i]), 4)
            for i in range(len(CLASS_NAMES))
        }
    }
    return response
