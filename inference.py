import codecs
import os
import warnings

import numpy as np
import torch
from poyo import parse_string
from torch.utils.data import DataLoader

from common_blocks.datasets import TestDataset
from common_blocks.transforms import get_transforms
from common_blocks.utils import create_folds
from models.lightningclassifier import LightningClassifier

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_tta_preds(net, images, augment=["null"], th=0.5):
    with torch.no_grad():
        net.eval()
        # import ipdb; ipdb.set_trace()
        if 1:  # null
            logit = net(images)
            probability = torch.sigmoid(logit)
        if "flip_lr" in augment:
            logit = net(torch.flip(images, dims=[3]))
            probability += torch.sigmoid(logit)
        if "flip_ud" in augment:
            logit = net(torch.flip(images, dims=[2]))
            probability += torch.sigmoid(logit)
        probability = probability / len(augment)
    return probability.data.cpu().numpy()


def get_all_models(path):
    all_models = []
    for model_path in os.listdir(path):
        model = LightningClassifier(config)
        checkpoint = torch.load(
            os.path.join(path, model_path), map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.freeze()
        all_models.append(model)
    return all_models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    folds = create_folds(config["validation"])
    dataset = TestDataset(
        folds[folds['label'] == 'correct'].head(25),
        config["test_inference"]["Dataset"],
        transform=get_transforms(data="valid", width=config["test_inference"]["Dataset"]["target_width"],
                                 height=config["test_inference"]["Dataset"]["target_height"]))
    loader = DataLoader(dataset, **config["test_inference"]["loader"])
    all_models = get_all_models(config["test_inference"]["models_path"])

    model_results = {"preds": [], "image_names": [], "image_label": {}}
    for fnames, images in loader:

        images = images.to(device)
        for model in all_models:
            batch_preds = None
            if batch_preds is None:
                batch_preds = get_tta_preds(model, images, augment=config["test_inference"]["TTA"])
            else:
                batch_preds += get_tta_preds(
                    model, images, augment=config["test_inference"]["TTA"]
                )
        model_results["image_names"].extend([i for i in fnames])

    model_results["preds"].append(batch_preds / len(all_models))
    model_results['preds'] = np.concatenate(model_results["preds"]).ravel()
    model_results["image_label"] = list((model_results["preds"] > config["test_inference"]["threshold"]
                                         ).astype(int))

    print(model_results)
    # folds = create_folds(config['validation'])
    # for fold in range(config['validation']['nfolds']):
    #     trn_idx = folds[folds['fold'] != fold].index
    #     val_idx = folds[folds['fold'] == fold].index
    #
    #     valid_dataset = TrainDataset(folds.loc[val_idx].reset_index(drop=True),
    #                                  config['Val']['Dataset'],
    #                                  transform=get_transforms(data='valid'))
    #     valid_loader = DataLoader(valid_dataset, **config['Val']['loader'])
    #
    #     model = LightningClassifier(config)
    #     checkpoint = torch.load('./lightning_logs/efficientnet_b2b/fold_1/epoch=06-avg_val_metric=0.9976.ckpt',
    #                             map_location=lambda storage, loc: storage)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     trainer = pl.Trainer()
    #     # todo fix returning results
    #     results
