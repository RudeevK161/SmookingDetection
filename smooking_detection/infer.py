import json
import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from pathlib import Path

from model import SmookingBinaryClassifier
from models.conv_net import ConvNet
from models.vit_model import ViTModel, ViTPreprocessor


class SmookingDetectionInference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(cfg)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.data.image_size, cfg.data.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=ViTPreprocessor.image_mean, std=ViTPreprocessor.image_std)
            ]
        )

    def _load_model(self, cfg: DictConfig):
        if cfg.model.model_name == "conv_net":
            base_model = ConvNet()
        elif "vit" in cfg.model.model_name.lower():
            model_str = getattr(cfg.model, "vit_model_name", 'WinKawaks/vit-tiny-patch16-224')
            id2label = getattr(cfg.model, "id2label", {0: "not_smoking", 1: "smoking"})
            label2id = getattr(cfg.model, "label2id", {"not_smoking": 0, "smoking": 1})
            if isinstance(id2label, DictConfig):
                id2label = OmegaConf.to_container(id2label, resolve=True)

            if isinstance(label2id, DictConfig):
                label2id = OmegaConf.to_container(label2id, resolve=True)

            base_model = ViTModel(
                model_name=model_str,
                id2label=id2label,
                label2id=label2id
            )
        else:
            raise ValueError(f"Unsupported model: {self.cfg.model.model_name}")

        model = SmookingBinaryClassifier(self.cfg)
        model.model = base_model

        checkpoint = torch.load(cfg.model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        input_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)

        not_smoking_prob = probs[0][0].item()
        smoking_prob = probs[0][1].item()

        if smoking_prob >= threshold:
            pred_idx = 1
            is_smoking = True
        else:
            pred_idx = 0
            is_smoking = False

        if hasattr(self, 'cfg') and hasattr(self.cfg.model, 'id2label'):
            id2label = getattr(self.cfg.model, "id2label", {0: "not_smoking", 1: "smoking"})
        elif hasattr(self, 'model') and hasattr(self.model, 'id2label'):
            id2label = self.model.id2label
        else:
            id2label = {0: "not_smoking", 1: "smoking"}

        if isinstance(id2label, DictConfig):
            id2label = OmegaConf.to_container(id2label, resolve=True)

        pred_class_name = id2label.get(pred_idx, f"class_{pred_idx}")
        confidence = max(not_smoking_prob, smoking_prob)

        return {
            "is_smoking": is_smoking,
            "predicted_class": pred_idx,
            "predicted_class_name": pred_class_name,
            "confidence": confidence,
            "threshold_used": threshold,
        }


@hydra.main(version_base=None, config_path="../config", config_name="config")
def detection_smooking(cfg: DictConfig):

    if not cfg.image_path:
        raise ValueError("image_path must be provided")

    if not cfg.model_path:
        default_paths = [
            "./export/model.pth",
            "./checkpoints/last.ckpt",
            "./checkpoints/best.ckpt"
        ]
        for path in default_paths:
            if Path(path).exists():
                cfg.model_path = path
                print(f"Using model from: {path}")
                break

        if not cfg.model_path:
            raise ValueError("model_path not provided and no default model found")

    classifier = SmookingDetectionInference(cfg)
    result = classifier.predict(cfg.image_path)

    print(f"Class: {result['predicted_class_name'].upper()}")
    print(f"Class ID: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Threshold used: {result['threshold_used']}")


if __name__ == "__main__":
    detection_smooking()