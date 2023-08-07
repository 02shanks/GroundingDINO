from groundingdino.models.GroundingDINO import GroundingDINO, build_transformer
from groundingdino.models.GroundingDINO.backbone import build_backbone
from groundingdino.util.utils import get_phrases_from_posmap
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from PIL import Image
import numpy as np
import torch
import requests
import bisect


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##text input
TEXT_PROMPT = "chair . person . dog ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

##load and transfrom image
transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
url = "https://images.ctfassets.net/4f3rgqwzdznj/44QovdzRYnjX4e0nv4lDJ8/1093abd8378d931144d71b16cbbdba1d/vet_hugs_cat_in_clinic_1332755026.jpg"
image_source = Image.open(requests.get(url, stream=True).raw).convert("RGB")
raw_image = np.asarray(image_source)
image_transformed, _ = transform(image_source, None)


##load model
model_config_path = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
model_checkpoint_url= "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
args = SLConfig.fromfile(model_config_path)
args.device = device

backbone = build_backbone(args)
transformer = build_transformer(args)

dn_labelbook_size = args.dn_labelbook_size
dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
sub_sentence_present = args.sub_sentence_present

model = GroundingDINO(
    backbone,
    transformer,
    num_queries=args.num_queries,
    aux_loss=True,
    iter_update=True,
    query_dim=4,
    num_feature_levels=args.num_feature_levels,
    nheads=args.nheads,
    dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
    two_stage_type=args.two_stage_type,
    two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
    two_stage_class_embed_share=args.two_stage_class_embed_share,
    num_patterns=args.num_patterns,
    dn_number=0,
    dn_box_noise_scale=args.dn_box_noise_scale,
    dn_label_noise_ratio=args.dn_label_noise_ratio,
    dn_labelbook_size=dn_labelbook_size,
    text_encoder_type=args.text_encoder_type,
    sub_sentence_present=sub_sentence_present,
    max_text_len=args.max_text_len,
)

checkpoint = torch.hub.load_state_dict_from_url(model_checkpoint_url, map_location="cpu")
model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
model.eval()

result = TEXT_PROMPT.lower().strip()
if result.endswith("."):
    caption = result
else:
    caption = result + "."

model = model.to(device)
image = image_transformed.to(device)

print(image.shape)

# for name, param in model.named_parameters():
#     print(name , param.shape)

pixel_values = image.unsqueeze(0) #"unsqueezeing" the tensor

with torch.no_grad():
    outputs = model.forward(samples=pixel_values, captions=[caption])

prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

mask = prediction_logits.max(dim=1)[0] > BOX_TRESHOLD
logits = prediction_logits[mask]  # logits.shape = (n, 256)
boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

tokenizer = model.tokenizer
tokenized = tokenizer(caption)
remove_combined = False
if remove_combined:
    sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
    
    phrases = []
    for logit in logits:
        max_idx = logit.argmax()
        insert_idx = bisect.bisect_left(sep_idx, max_idx)
        right_idx = sep_idx[insert_idx]
        left_idx = sep_idx[insert_idx - 1]
        phrases.append(get_phrases_from_posmap(logit > TEXT_TRESHOLD, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
else:
    phrases = [
        get_phrases_from_posmap(logit > TEXT_TRESHOLD, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

print("phrases", phrases)   
print("boxes", boxes)
print("logits", prediction_logits.shape, logits)