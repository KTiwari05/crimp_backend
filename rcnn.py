import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    num_classes_from_model = checkpoint["roi_heads.box_predictor.cls_score.weight"].shape[0]
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes_from_model)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model

def transform_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def predict(model, image_tensor, device):
    model.to(device)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        predictions = model([image_tensor])
    return predictions

def draw_predictions(image_path, predictions, labels_map, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    for i in range(len(predictions[0]["boxes"])):
        box = predictions[0]["boxes"][i].cpu().numpy().astype(int)
        score = predictions[0]["scores"][i].cpu().numpy()
        defect_label_id = int(predictions[0]["labels"][i].cpu().numpy())
        defect_label = labels_map.get(defect_label_id, "Other Defect")
        if score > threshold:
            cv2.rectangle(image_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
            cv2.putText(image_np, f"{defect_label}", (box[0], box[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    plt.figure()
    plt.imshow(image_np)
    plt.axis("off")
    plt.show()

def main(image_path):
    checkpoint_path = r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\crimp_defect_detector 1.pth"
    labels_map = {
        0: "arivoid",
        1: "Conductor Show",
        2: "Overlapping wings",
        3: "Terminal sheet cracked",
        4: "Wing folding back",
        5: "Wings Dissymmetry",
        6: "Wings are not in contact",
        7: "Wings touching terminal floor or wall",
        8: "Wrong cross section uploaded",
        9: "arivoid",
        10: "burr",
        11: "ok crimp"
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path)
    image_tensor = transform_image(image_path)
    predictions = predict(model, image_tensor, device)
    draw_predictions(image_path, predictions, labels_map)

if __name__ == "__main__":
    image_path = r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\airlab\examples\Input\image1.JPG"  # Replace with your image path
    main(image_path)
