import torch
from PIL import Image
from torchvision import transforms
from src.models.ensemble import CombinedModel

def predict(model_path, img_path, class_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(n_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    tfm = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = torch.softmax(model(x), dim=1)
        cls = preds.argmax(1).item()
    return class_names[cls], preds[0][cls].item()

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model-path', type=str, required=True)
    p.add_argument('--img', type=str, required=True)
    args = p.parse_args()
    print(predict(args.model_path, args.img, ['class0','class1','class2']))
