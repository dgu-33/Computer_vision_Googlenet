"""
Inference script for a trained GoogLeNet model on the POC dataset.
"""

import os
import torch
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
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

    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with probability bar chart."""
        predicted_class, confidence, all_probs = self.predict_single_image(image_path)
        image = Image.open(image_path)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].axis('off')
        axes[0].set_title(
            f'Predicted: {predicted_class}\nConfidence: {confidence:.2%}',
            fontsize=14, fontweight='bold'
        )

        colors = [
            'green' if i == np.argmax(all_probs) else 'skyblue'
            for i in range(len(self.class_names))
        ]
        axes[1].barh(self.class_names, all_probs, color=colors)
        axes[1].set_xlim([0, 1])
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Class Probabilities', fontsize=14)

        for i, (prob, _) in enumerate(zip(all_probs, self.class_names)):
            axes[1].text(prob + 0.01, i, f'{prob:.2%}', va='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()
        return predicted_class, confidence

    def predict_batch(self, image_folder):
        """Run inference on all images in a folder."""
        image_files = [
            f for f in os.listdir(image_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]

        results = []
        print(f"Found {len(image_files)} images in {image_folder}")

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

        return results


def main():
    MODEL_PATH = "models/best_googlenet_poc.pth"
    TEST_IMAGE = "POC_Dataset/Testing/Chorionic_villi/test_image.jpg"
    TEST_FOLDER = "POC_Dataset/Testing/Chorionic_villi"

    predictor = POCPredictor(model_path=MODEL_PATH)

    if os.path.exists(TEST_IMAGE):
        print("\n" + "="*50)
        print("Single Image Prediction")
        print("="*50 + "\n")

        pred_class, conf = predictor.visualize_prediction(
            TEST_IMAGE,
            save_path="prediction_result.png"
        )
        print(f"\nPrediction: {pred_class}")
        print(f"Confidence: {conf:.2%}")

    if os.path.exists(TEST_FOLDER):
        print("\n" + "="*50)
        print("Batch Prediction")
        print("="*50 + "\n")

        results = predictor.predict_batch(TEST_FOLDER)

        print("\n" + "="*50)
        print("Summary")
        print("="*50)
        print(f"Total images: {len(results)}")
        print(f"Avg confidence: {np.mean([r['confidence'] for r in results]):.2%}")


if __name__ == "__main__":
    main()
