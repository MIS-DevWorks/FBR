from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import sys

from models.MFA_ViT import MFA_ViT
from utils.dataset import VGGFace2_dataset, custom_multimodal_dataset
from utils.loss import total_LargeMargin_CrossEntropy, CFPC_loss
from utils.model_utils import *
import config


def training(args):
    """
        Training function for MFA-ViT.

        :param args: Hyperparameter configurations
    """
    cur_acc_dict = {
        "face": 0.,
        "ocu": 0.,
    }
    writer = SummaryWriter(log_dir=config.save_dir + '/summary')
    os.makedirs(config.save_dir + "/checkpoints", exist_ok=True)

    # Dataset
    print("-----------------------------------------------------------------------------")
    if args.dataset_name == "VGGFace2" or args.dataset_name is None:
        print("\tDataset Loading: " + str(args.dataset_name))
        train_dataset = VGGFace2_dataset(csv_path=config.dataset_path_csv_train_path,
                                         img_path=config.dataset_path_img_path, train_augmentation=True,
                                         image_size=config.image_size)
        valid_dataset = VGGFace2_dataset(csv_path=config.dataset_path_csv_test_path,
                                         img_path=config.dataset_path_img_path,
                                         train_augmentation=False, image_size=config.image_size, biometric_mode="multi")
    else:
        print("\tNo dataset is selected for training.")
        print("-----------------------------------------------------------------------------\n\n")
        sys.exit(0)
    print("-----------------------------------------------------------------------------\n\n")

    print("-----------------------------------------------------------------------------")
    print("\tMFA-ViT Model\n")
    print("\tConfiguration Settings")
    print("\t\tInput Size (face and periocular): %d" % config.image_size)
    print("\t\tPatch Embedding Size: %d" % config.patch_size)
    print("\t\tLayer Depth: %d" % config.layer_depth)
    print("\t\tNumber of Head: %d" % config.num_heads)
    print("\t\tPrompt Strategy: %s" % config.prompt_mode)
    print("\t\tSize of Prompt Embeddings: %d" % config.prompt_tokens)
    print("\t\tClassification Head Input: %s" % config.head_strategy)
    # Model
    model = MFA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                    embed_dim=config.embed_dim, num_classes=config.num_sub_classes, layer_depth=config.layer_depth,
                    num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, norm_layer=None,
                    drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate, prompt_mode=config.prompt_mode,
                    prompt_tokens=config.prompt_tokens, head_strategy=config.head_strategy)

    if config.multigpu:
        print("\t\tUsing Multi-gpu")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)

    if config.pretrained_weights is not None:
        print("\t\tLoading pre-trained weights")
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device), strict=True)

    model = model.to(config.device)

    # Optimizer
    if config.optimizer == "SGD":
        print("\t\tOptimizer: SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=config.init_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
    elif config.optimizer == "Adam":
        print("\t\tOptimizer: Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)
    elif config.optimizer == "AdamW":
        print("\t\tOptimizer: AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay)

    # Loss functions
    loss_subj = total_LargeMargin_CrossEntropy().to(config.device)
    loss_cm1 = CFPC_loss(temperature=config.temperature, negative_weight=config.negative_weight,
                         config=config).to(config.device)
    loss_cm2 = CFPC_loss(temperature=config.temperature_other, negative_weight=config.negative_weight,
                         config=config).to(config.device)
    loss_cm3 = CFPC_loss(temperature=config.temperature_other, negative_weight=config.negative_weight,
                         config=config).to(config.device)

    # Training
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_data_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    print("-----------------------------------------------------------------------------\n\n")

    print("-----------------------------------------------------------------------------")
    print("\tTraining...")
    for epoch in range(config.start_epoch, config.total_epochs):
        print("[EPOCH {}/{}]".format(epoch, config.total_epochs - 1))
        train_loss, train_acc_dict = train(data_loader=train_data_loader, model=model,
                                           lm_loss_fn=loss_subj, cm_loss_fn_list=[loss_cm1, loss_cm2, loss_cm3],
                                           optimizer=optimizer, epoch=epoch, writer=writer, config=config)
        valid_loss, valid_acc_dict = valid(data_loader=valid_data_loader, model=model,
                                           lm_loss_fn=loss_subj, cm_loss_fn=loss_cm1,
                                           epoch=epoch, writer=writer, config=config)

        if valid_acc_dict["ocu"] >= cur_acc_dict["ocu"]:
            cur_acc_dict = valid_acc_dict
            torch.save(model.state_dict(), "{}/checkpoints/best_model.pt".format(config.save_dir))
        torch.save(model.state_dict(), "{}/checkpoints/latest_model.pt".format(config.save_dir))

        print("    [Train] loss: {:.4f}, \nacc: {}\n".format(train_loss, train_acc_dict))
        print("    [Valid] loss: {:.4f},\nacc: {}\n".format(valid_loss, valid_acc_dict))
        print("    [BEST] acc: {}".format(cur_acc_dict))
    print("-----------------------------------------------------------------------------\n")
        

def evaluation(args):
    """
        Evaluation function for MFA-ViT.

        :param args: Hyperparameter configurations
    """
    print("-----------------------------------------------------------------------------")
    print("\tLoading evaluation dataset")
    if args.dataset_name == "custom" or args.dataset_name == "other":
        gallery_dataset = custom_multimodal_dataset(config, subdir="rgb/", train_augmentation=False,
                                                    imagesize=config.image_size)
        gallery_data_loader = DataLoader(gallery_dataset, batch_size=config.batch_size, shuffle=False,
                                         num_workers=config.num_workers)
        probe_dataset = custom_multimodal_dataset(config, subdir="rgb/", train_augmentation=False,
                                                  imagesize=config.image_size)
        probe_data_loader = DataLoader(probe_dataset, batch_size=config.batch_size, shuffle=False,
                                       num_workers=config.num_workers)
    print("-----------------------------------------------------------------------------\n\n")

    print("-----------------------------------------------------------------------------")
    print("\tMFA-ViT Model\n")
    print("\tCofiguration Settings")
    print("\t\tInput Size (face and periocular): %d" % config.image_size)
    print("\t\tPatch Embedding Size: %d" % config.patch_size)
    print("\t\tLayer Depth: %d" % config.layer_depth)
    print("\t\tNumber of Head: %d" % config.num_heads)
    print("\t\tPrompt Strategy: %s" % config.prompt_mode)
    print("\t\tSize of Prompt Embeddings: %d" % config.prompt_tokens)
    print("\t\tClassitiona Head Input: %s" % config.head_strategy)

    # Model
    model = MFA_ViT(img_size=config.image_size, patch_size=config.patch_size, in_chans=config.in_chans,
                    embed_dim=config.embed_dim, num_classes=config.num_sub_classes,
                    layer_depth=config.layer_depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
                    norm_layer=None, drop_rate=config.drop_rate, attn_drop_rate=config.attn_drop_rate,
                    drop_path_rate=config.drop_path_rate, prompt_mode=config.prompt_mode,
                    prompt_tokens=config.prompt_tokens, head_strategy=config.head_strategy)
    if config.multigpu:
        print("\t\tUsing Multi-GPU")
        model = torch.nn.DataParallel(model, device_ids=config.device_ids)
    model = model.to(config.device)

    if config.pretrained_weights is not None:
        print("\t\tUsing pre-trained weights")
        model.load_state_dict(torch.load(config.pretrained_weights, map_location=config.device), strict=True)
    print("-----------------------------------------------------------------------------\n\n")

    print("-----------------------------------------------------------------------------")
    print("\tEvaluation on progress...")
    gallery_data_features_dict = get_features(model, gallery_data_loader, config)
    probe_data_features_dict = get_features(model, probe_data_loader, config)

    ''' gallery face vs probe face '''
    face_face_acc_by_max = evaluate_crossmodal_data_features_dict(gallery_data=gallery_data_features_dict[0],
                                                                  probe_data=probe_data_features_dict[0],
                                                                  gallery_gt=gallery_data_features_dict[2],
                                                                  probe_gt=probe_data_features_dict[2],
                                                                  method='max')
    print("\t\t[TEST] face2face acc by cos_sim - max    : {}".format(face_face_acc_by_max))

    ''' gallery ocu vs probe ocu '''
    ocu_ocu_acc_by_max = evaluate_crossmodal_data_features_dict(gallery_data=gallery_data_features_dict[1],
                                                                probe_data=probe_data_features_dict[1],
                                                                gallery_gt=gallery_data_features_dict[2],
                                                                probe_gt=probe_data_features_dict[2],
                                                                method='max')
    print("\t\t[TEST] ocu2ocu acc by cos_sim - max    : {}".format(ocu_ocu_acc_by_max))

    ''' gallery face vs probe ocu '''
    face_ocu_acc_by_max = evaluate_crossmodal_data_features_dict(gallery_data=gallery_data_features_dict[0],
                                                                 probe_data=probe_data_features_dict[1],
                                                                 gallery_gt=gallery_data_features_dict[2],
                                                                 probe_gt=probe_data_features_dict[2],
                                                                 method='max')
    print("\t\t[TEST] face2ocu acc by cos_sim - max    : {}".format(face_ocu_acc_by_max))

    ''' gallery ocu vs probe face '''
    ocu_face_acc_by_max = evaluate_crossmodal_data_features_dict(gallery_data=gallery_data_features_dict[1],
                                                                 probe_data=probe_data_features_dict[0],
                                                                 gallery_gt=gallery_data_features_dict[2],
                                                                 probe_gt=probe_data_features_dict[2],
                                                                 method='max')
    print("\t\t[TEST] ocu2face acc by cos_sim - max    : {}".format(ocu_face_acc_by_max))
    print("-----------------------------------------------------------------------------")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_mode', help='Training mode', action='store_true')
    parser.add_argument('--dataset_name', help='Dataset name', default=None)

    return parser.parse_args(argv)


def main(args):
    if args.training_mode is True:
        training(args)
    elif args.training_mode is False:
        evaluation(args)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
    sys.exit(0)
