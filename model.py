import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import numpy as np


class ResNetBackbone(nn.Module):
    def __init__(self, model_name="resnet", hidden_size=768, num_answers=1000):
        super(ResNetBackbone, self).__init__()
        
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  
        self.text_embed = nn.Embedding(10000, 300)  
        self.text_encoder = nn.GRU(300, hidden_size, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2048 + hidden_size, hidden_size),nn.ReLU(),nn.Dropout(0.5))
        self.classifier = nn.Linear(hidden_size, num_answers)
    
    def forward(self, images, questions=None):
        batch_size = images.shape[0]
        visual_features = self.backbone(images) 
        
        self.feature_maps = visual_features.clone()
        visual_features = visual_features.mean(dim=(2, 3))  
        
        text_tokens = torch.ones(batch_size, 10).long().to(images.device)  
        text_embeds = self.text_embed(text_tokens)
        _, text_features = self.text_encoder(text_embeds)
        text_features = text_features.squeeze(0)  
        
        multimodal_features = self.fusion(torch.cat([visual_features, text_features], dim=1))    
        logits = self.classifier(multimodal_features)
        
        return {
            "logits": logits,
            "hidden_states": [multimodal_features],  
            "last_hidden_state": multimodal_features,
            "feature_maps": self.feature_maps  
        }

class AdvancedAnswerGroundingModel(nn.Module):
    def __init__(
            self,
            vqa_model=None,
            pretrained=True,
            freeze_backbone=True,
            hidden_size=768,
            num_answers=1000,
            model_name="resnet"
    ):
        super(AdvancedAnswerGroundingModel, self).__init__()
    
        self.vqa_model = ResNetBackbone(
            model_name=model_name,
            hidden_size=hidden_size,
            num_answers=num_answers
        )
        self.feature_maps = None

        self.obj_detector = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        if freeze_backbone:
            for param in self.obj_detector.parameters():
                param.requires_grad = False
        
        self.feature_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1)
        )
        
        self.object_attention = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),  
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.attention_fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        self.refinement = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, images, questions=None):
        vqa_outputs = self.vqa_model(images, questions)
        if hasattr(self.vqa_model, 'feature_maps'):
            self.feature_maps = self.vqa_model.feature_maps

        # Feature llevel attention
        feature_attn = self.feature_attention(self.feature_maps)
        feature_attn = torch.sigmoid(feature_attn)
        
        # object level attention
        original_images = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            original_images.append(torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0)
            
        obj_attn_maps = []
        self.obj_detector.eval()
        self.obj_detector.to(images.device) 

        with torch.no_grad():
            for img in original_images:
                predictions = self.obj_detector([img.to(images.device)])
                obj_attn = torch.zeros((1, 224, 224), device=images.device)
                boxes = predictions[0]['boxes']
                scores = predictions[0]['scores']
                
                keep = scores > 0.5
                boxes = boxes[keep]
                scores = scores[keep]
                
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box.int()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(223, x2), min(223, y2)
                    obj_attn[0, y1:y2, x1:x2] += score
                
                obj_attn_maps.append(obj_attn)
            
        obj_attn = torch.stack(obj_attn_maps)
        
        # fusse attention maps
        feature_attn_up = self.upsample(feature_attn)
        obj_attn = obj_attn.unsqueeze(1) if obj_attn.dim() == 3 else obj_attn
        
        if obj_attn.shape != feature_attn_up.shape:
            obj_attn = F.interpolate(obj_attn, size=feature_attn_up.shape[2:], mode='bilinear', align_corners=False)
            obj_attn = obj_attn.to(feature_attn_up.device)
        
        combined_attn = torch.cat([feature_attn_up, obj_attn], dim=1)
        fused_attn = self.attention_fusion(combined_attn)

        refined_input = torch.cat([fused_attn, images], dim=1)
        mask_probs = self.refinement(refined_input)
        
        mask_logits = mask_probs  
        
        return {
            "logits": vqa_outputs["logits"],
            "mask_logits": mask_logits,
            "mask_probs": mask_probs,
            "feature_attention": feature_attn_up,
            "object_attention": obj_attn
        }