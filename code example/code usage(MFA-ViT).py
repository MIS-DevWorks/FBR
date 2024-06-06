import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.'))
sys.path.append(parent_dir)

import torch
import config
from models.MFA_ViT import MFA_ViT, model_params


DEVICE = "cuda"
print("-----------------------------------------------------------------------------")
print("\tConfiguration Settings")
print("\t\tInput Size (face and periocular): %d" % config.image_size)
print("\t\tPatch Embedding Size: %d" % config.patch_size)
print("\t\tLayer Depth: %d" % config.layer_depth)
print("\t\tNumber of Head: %d" % config.num_heads)
print("\t\tPrompt Strategy: %s" % config.prompt_mode)
print("\t\tSize of Prompt Embeddings: %d" % config.prompt_tokens)
print("\t\tClassification Head Input: %s" % config.head_strategy)


print("\n\tRunning test code: MFA-ViT")
input1 = torch.rand(1, 1, 3, 112, 112).to(DEVICE)  # face
input2 = torch.rand(1, 2, 3, 112, 112).to(DEVICE)  # ocular
input3 = torch.randint(low=0, high=2, size=(1, 47)).to(DEVICE)  # attribute
model = MFA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                embed_dim=config.embed_dim, num_classes=config.num_sub_classes,
                layer_depth=config.layer_depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                norm_layer=config.norm_layer, drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                drop_path_rate=config.drop_path_rate, prompt_mode=config.prompt_mode,
                prompt_tokens=config.prompt_tokens, head_strategy=config.head_strategy).to(DEVICE)
print("\t\tTotal Params: {:.2f}M".format(model_params(model) / 1000000))
print("-----------------------------------------------------------------------------")

y = model(input1, input2, input3, return_feature=True)

sys.exit(0)
