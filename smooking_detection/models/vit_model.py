import torch.nn as nn
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


class ViTModel(nn.Module):
    def __init__(self, model_name='WinKawaks/vit-tiny-patch16-224',
                 id2label=None, label2id=None):
        super().__init__()

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        if hasattr(self.model, 'classifier'):
            if isinstance(self.model.classifier, nn.Linear):
                self.model.classifier = nn.Linear(
                    self.model.classifier.in_features,
                    2
                )

        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            inputs = self.processor(images=x, return_tensors="pt")
            x = inputs["pixel_values"]

        outputs = self.model(x)

        return outputs.logits if hasattr(outputs, 'logits') else outputs


class ViTPreprocessor:

    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    def __init__(self, model_name="WinKawaks/vit-tiny-patch16-224"):
        self.processor = ViTImageProcessor.from_pretrained(model_name)

    def __call__(self, images):
        return self.processor(images, return_tensors="pt")