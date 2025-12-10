"""
Inference script for a trained GoogLeNet model on the POC dataset.
"""

import os
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from poc_dataset import get_data_transforms


class POCPredictor:
    def __init__(self, model_path, num_classes=4, device=None):
        self.num_classes = num_classes
        self.class_names = [
            'Chorionic_villi',
            'Decidual_tissue',
            'Hemorrhage',
            'Trophoblastic_tissue'
        ]
        
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None else device
        )
        
        self.model = self._load_model(model_path)
        self.model.eval()
        self.transform = get_data_transforms(is_training=False)
        
        print(f"Model loaded on {self.device}")

    def _load_model(self, model_path):
        """Load model architecture and trained weights."""
        model = models.googlenet(pretrained=False, aux_logits=True)
        
        # Replace final layers (must match training configuration)
        model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)

        if hasattr(model, 'aux1'):
            model.aux1.fc2 = torch.nn.Linear(model.aux1.fc2.in_features, self.num_classes)
        if hasattr(model, 'aux2'):
            model.aux2.fc2 = torch.nn.Linear(model.aux2.fc2.in_features, self.num_classes)

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(self.device)

    def predict_single_image(self, image_path):
        """Predict class probabilities for a single image."""
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            if isinstance(outputs, tuple): 
                outputs = outputs[0]

            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return (
            self.class_names[predicted.item()],
            confidence.item(),
            probs[0].cpu().numpy()
        )

    def predict_batch(self, image_folder, true_label):
        """
        Run inference on all images in a folder and collect predictions.
        true_label: integer index of the ground truth class
        """
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        results = []
        y_true = []
        y_pred = []

        print(f"\nFound {len(image_files)} images in {image_folder}")

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            pred_class, conf, all_probs = self.predict_single_image(img_path)

            results.append({
                'filename': img_file,
                'predicted_class': pred_class,
                'confidence': conf,
                'probabilities': {
                    name: prob for name, prob in zip(self.class_names, all_probs)
                }
            })

            print(f"{img_file}: {pred_class} ({conf:.2%})")

            # Metrics 계산용
            y_true.append(true_label)
            y_pred.append(self.class_names.index(pred_class))

        return results, y_true, y_pred


def main():
    MODEL_PATH = "models/best_googlenet_poc.pth"
    BASE_TEST_DIR = "POC_Dataset/Testing"

    predictor = POCPredictor(model_path=MODEL_PATH)

    all_true = []
    all_pred = []

    print("\n===============================================")
    print("Batch Prediction for All Classes")
    print("===============================================\n")

    # iterate each folder
    for class_idx, class_name in enumerate(predictor.class_names):
        test_folder = os.path.join(BASE_TEST_DIR, class_name)

        if not os.path.exists(test_folder):
            print(f"❌ Folder not found: {test_folder}")
            continue

        print(f"\n--- Predicting folder: {class_name} ---\n")

        results, y_true, y_pred = predictor.predict_batch(
            test_folder,
            true_label=class_idx
        )

        all_true.extend(y_true)
        all_pred.extend(y_pred)

    print("\n===============================================")
    print("Classification Report")
    print("===============================================\n")

    print(classification_report(
        all_true, 
        all_pred, 
        target_names=predictor.class_names,
        digits=4
    ))

    print("\nOverall Accuracy:", accuracy_score(all_true, all_pred))


if __name__ == "__main__":
    main()
