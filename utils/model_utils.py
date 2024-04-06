from tqdm import tqdm
import torch


def train(data_loader, model, lm_loss_fn, cm_loss_fn_list, optimizer, epoch, writer, config):
    """
        Training function.

        :param data_loader: PyTorch's data loader format
        :param model: PyTorch's model structure
        :param lm_loss_fn: Large-margin loss function
        :param cm_loss_fn_list: Cross-modality loss functions in list format
        :param optimizer: Optimizer function
        :param epoch: Current epoch index
        :param writer: Saving information in Tensorboard format
        :param config: Configuration file -> config.py
        :return: Total of training loss value and a dictionary of accuracy data
    """
    model.train()
    total_loss_list, total_cm_loss1_list, total_cm_loss2_list, total_cm_loss3_list, total_lm_loss_list = [], [], [], [], []
    total = 0
    iter_dict = {}

    for i, (face_input, ocu_input, attr_input, target) in enumerate(tqdm(data_loader)):
        face_fea, ocu_fea, attr_fea, face_pred, ocu_pred = \
            model.forward(face_input.to(config.device), ocu_input.to(config.device), 
                          attr_input.to(config.device), return_feature=True)

        target = target.to(config.device)
        cm_loss1 = cm_loss_fn_list[0](face_fea, ocu_fea)
        cm_loss2 = cm_loss_fn_list[1](face_fea, attr_fea)
        cm_loss3 = cm_loss_fn_list[2](ocu_fea, attr_fea)
        lm_loss = lm_loss_fn(s1=face_pred, s2=ocu_pred, target=target)
        total_l = lm_loss + cm_loss1 + cm_loss2 + cm_loss3
        
        total_cm_loss1_list.append(cm_loss1)
        total_cm_loss2_list.append(cm_loss2)
        total_cm_loss3_list.append(cm_loss3)
        total_lm_loss_list.append(lm_loss)
        total_loss_list.append(total_l)
        total += target.size(0)

        optimizer.zero_grad()
        total_l.backward()
        optimizer.step()

        if i == 0:  # first iteration
            iter_dict["actual"] = target
            iter_dict["face"] = torch.argmax(face_pred, dim=1)
            iter_dict["ocu"] = torch.argmax(ocu_pred, dim=1)
        else:
            iter_dict["actual"] = torch.cat((iter_dict["actual"], target))
            iter_dict["face"] = torch.cat((iter_dict["face"], torch.argmax(face_pred, dim=1)))
            iter_dict["ocu"] = torch.cat((iter_dict["ocu"], torch.argmax(ocu_pred, dim=1)))

    acc_dict = {"face": torch.sum(iter_dict["face"] == iter_dict["actual"]).item() / total,
                "ocu": torch.sum(iter_dict["ocu"] == iter_dict["actual"]).item() / total}

    for key in acc_dict:
        writer.add_scalar("train/{}_acc".format(key), acc_dict[key], epoch)

    total_cm_loss1 = torch.mean(torch.tensor(total_cm_loss1_list))
    total_cm_loss2 = torch.mean(torch.tensor(total_cm_loss2_list))
    total_cm_loss3 = torch.mean(torch.tensor(total_cm_loss3_list))
    total_lm_loss = torch.mean(torch.tensor(total_lm_loss_list))
    total_loss = torch.mean(torch.tensor(total_loss_list))
    writer.add_scalar("train/total_cm_loss1", total_cm_loss1, epoch)
    writer.add_scalar("train/total_cm_loss2", total_cm_loss2, epoch)
    writer.add_scalar("train/total_cm_loss3", total_cm_loss3, epoch)
    writer.add_scalar("train/total_lm_loss", total_lm_loss, epoch)
    writer.add_scalar("train/total_loss", total_loss, epoch)

    return total_loss, acc_dict


def valid(data_loader, model, lm_loss_fn, cm_loss_fn, epoch, writer, config):
    """
        Cross-validation function.

        :param data_loader: PyTorch's data loader format
        :param model: PyTorch's model structure
        :param lm_loss_fn: Large-margin loss function
        :param cm_loss_fn: Cross-modality loss function for face and periocular
        ::param epoch: Current epoch index
        :param writer: Saving information in Tensorboard format
        :param config: Configuration file -> config.py
        :return: Total of validation loss value and a dictionary of accuracy data
    """
    model.eval()
    total_loss_list, total_cm_loss_list, total_lm_loss_list = [], [], []
    total = 0
    iter_dict = {}

    for i, (face_input, ocu_input, target) in enumerate(tqdm(data_loader)):
        if i == 5:
            break
        with torch.no_grad():
            face_fea, ocu_fea, face_pred, ocu_pred = \
                model.forward(face_input.to(config.device), ocu_input.to(config.device), 
                              None, return_feature=True)

            target = target.to(config.device)
            cm_loss = cm_loss_fn(face_fea, ocu_fea)
            lm_loss = lm_loss_fn(s1=face_pred, s2=ocu_pred, target=target)
            total_l = lm_loss + cm_loss
            
            total_cm_loss_list.append(cm_loss)
            total_lm_loss_list.append(lm_loss)
            total_loss_list.append(total_l)
            total += target.size(0)

        if i == 0:  # first iteration
            iter_dict["actual"] = target
            iter_dict["face"] = torch.argmax(face_pred, dim=1)
            iter_dict["ocu"] = torch.argmax(ocu_pred, dim=1)
        else:
            iter_dict["actual"] = torch.cat((iter_dict["actual"], target))
            iter_dict["face"] = torch.cat((iter_dict["face"], torch.argmax(face_pred, dim=1)))
            iter_dict["ocu"] = torch.cat((iter_dict["ocu"], torch.argmax(ocu_pred, dim=1)))

    acc_dict = {"face": torch.sum(iter_dict["face"] == iter_dict["actual"]).item() / total,
                "ocu": torch.sum(iter_dict["ocu"] == iter_dict["actual"]).item() / total}
    for key in acc_dict:
        writer.add_scalar("valid/{}_acc".format(key), acc_dict[key], epoch)

    total_cm_loss = torch.mean(torch.tensor(total_cm_loss_list))
    total_lm_loss = torch.mean(torch.tensor(total_lm_loss_list))
    total_loss = torch.mean(torch.tensor(total_loss_list))
    writer.add_scalar("valid/total_cm_loss", total_cm_loss, epoch)
    writer.add_scalar("valid/total_lm_loss", total_lm_loss, epoch)
    writer.add_scalar("valid/total_loss", total_loss, epoch)

    return total_loss, acc_dict


def get_features(model, data_loader, config, prompt=True):
    """
        Extract biometric feature embeddinges.

        :param model: PyTorch's model structure
        :param data_loader: PyTorch's data loader format
        :param config: Configuration file -> config.py
        :param prompt: Using prompt strategy, where default value is True
        :return: gallery feature embedding, probe feature embedding, identity info
    """
    model.eval()
    subject_ids = []

    for i, (inputs, inputs2, targets) in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            if config.multigpu is True:
                gallery_features = model.module.tokenize(inputs.to(config.device), mode="face")
                if prompt is False:
                    gallery_features = model.module.forward_features(gallery_features)[:, 0, :]
                else:
                    gallery_features = model.module.forward_features(gallery_features, mode="face")[:, :]

                probe_features = model.module.tokenize(inputs2.to(config.device), mode="ocular")
                if prompt is False:
                    probe_features = model.module.forward_features(probe_features)[:, 0, :]
                else:
                    probe_features = model.module.forward_features(probe_features, mode="ocular")[:, :]
            else:
                gallery_features = model.forward_patch_embed(inputs.to(config.device), mode="face")
                if prompt is False:
                    gallery_features = model.forward_features(gallery_features)[:, 0, :]
                elif prompt is True:
                    gallery_features = model.forward_features(gallery_features, mode="face")[:, :]

                probe_features = model.forward_patch_embed(inputs2.to(config.device), mode="ocular")
                if prompt is False:
                    probe_features = model.forward_features(probe_features)[:, 0, :]
                elif prompt is True:
                    probe_features = model.forward_features(probe_features, mode="ocular")[:, :]

        subject_ids.extend(targets)
        for j, feature in enumerate(gallery_features):
            if i == 0 and j == 0:
                gallery_features_tensor = torch.unsqueeze(feature.detach(), dim=0)
            else:
                gallery_features_tensor = torch.cat(
                    (gallery_features_tensor, torch.unsqueeze(feature.detach(), dim=0)), dim=0)

        for j, feature in enumerate(probe_features):
            if i == 0 and j == 0:
                probe_features_tensor = torch.unsqueeze(feature.detach(), dim=0)
            else:
                probe_features_tensor = torch.cat(
                    (probe_features_tensor, torch.unsqueeze(feature.detach(), dim=0)), dim=0)

    return gallery_features_tensor, probe_features_tensor, torch.tensor(subject_ids)


def evaluate_crossmodal_data_features_dict(gallery_data, probe_data, gallery_gt, probe_gt):
    """
        Evaluation on cross-modality

        :param gallery_data: Gallery set
        :param probe_data: Probe set
        :param gallery_gt: Gallery ground truth set
        :param probe_gt: Probe ground truth set

        :return: Accuracy
    """
    total_true_preds = 0
    cos_sim_score_list = []
    for i, test_data_feature in enumerate(probe_data):
        cos_sim_score_list.append(torch.nn.functional.cosine_similarity(
            test_data_feature.repeat(len(gallery_data), 1), gallery_data))

    for i, cos_sim_score in enumerate(tqdm(cos_sim_score_list)):
        if gallery_gt[torch.argmax(cos_sim_score).item()] == probe_gt[i]:
            total_true_preds += 1

    return (total_true_preds / len(probe_data)) * 100.
