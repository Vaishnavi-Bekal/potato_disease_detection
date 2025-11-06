# Potato Leaf Disease Detection

A PyTorch-based deep learning project combining **VGG16** and **MobileNetV2** pretrained backbones to classify potato leaf diseases.

## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --data-dir ./data --epochs 10
python src/infer.py --model-path ./best.pth --img path/to/image.jpg
```

## API
```bash
 python -m uvicorn src.api.app:app --host localhost --port 8080 
```
