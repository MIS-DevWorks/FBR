from datetime import datetime


#   Path
dataset_path_csv_train_path             = 'data/MAAD_Face_filtered_train.csv'
dataset_path_csv_test_path              = 'data/MAAD_Face_filtered_valid.csv'
dataset_path_img_path                   = 'VGGFace2/train/'
other_dataset_path_img                  = 'Sample/'
pretrained_weights                      = 'pretrained/MFA-ViT.pt'
save_dir                                = 'save/MFA-ViT_version_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))


#   Image configuration
image_size                              = 112
in_chans                                = 3
num_sub_classes                         = 9131


#   Model configuration
device                                  = "cuda"
multigpu                                = True
device_ids                              = [0, ]
patch_size                              = 8
embed_dim                               = 1024
layer_depth                             = 4
num_heads                               = 12
mlp_ratio                               = 4.
norm_layer                              = None
drop_rate                               = 0.1
attn_drop_rate                          = 0.
drop_path_rate                          = 0.1
prompt_mode                             = "deep"
prompt_tokens                           = 32
head_strategy                           = "prm"


#   Training configuration
batch_size                              = 64
init_lr                                 = 1e-4
optimizer                               = "AdamW"
momentum                                = 0.9  # for SGD optimizer
weight_decay                            = 5e-4
start_epoch                             = 0
total_epochs                            = 50
temperature                             = 0.03
temperature_other                       = 0.04
negative_weight                         = 0.8
num_workers                             = 4
