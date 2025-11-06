import torch
from PIL import Image
from torchvision import transforms
from src.models.ensemble import CombinedModel

def predict(model_path, img_path, class_names, threshold=0.9):
    """
    Predicts the class of an input image using the trained ensemble model.
    If confidence < threshold, returns 'Unknown'.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(n_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    # Preprocessing
    tfm = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load and prepare image
    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        preds = torch.softmax(model(x), dim=1)
        cls = preds.argmax(1).item()
        confidence = preds[0][cls].item()

    # Apply confidence threshold
    if confidence < threshold:
        predicted_label = "Unknown"
    else:
        predicted_label = class_names[cls]

    return predicted_label, confidence


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Potato Disease Classifier with Confidence Threshold")
    p.add_argument('--model-path', type=str, required=True, help='Path to trained model (.pth)')
    p.add_argument('--img', type=str, required=True, help='Path to input image')
    p.add_argument('--threshold', type=float, default=0.9, help='Confidence threshold for Unknown class')
    args = p.parse_args()

    # 🟢 Replace class names with actual labels from your dataset
    class_names = ['Early Blight', 'Late Blight', 'Healthy']

    label, conf = predict(args.model_path, args.img, class_names, args.threshold)
    print(f"Predicted: {label} (Confidence: {conf*100:.2f}%)")
