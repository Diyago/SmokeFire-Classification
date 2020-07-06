import codecs
import os
import warnings

import numpy as np
import pandas as pd
import torch
from poyo import parse_string
from sklearn import metrics
from torch.utils.data import DataLoader

from common_blocks.datasets import TestDataset
from common_blocks.transforms import get_transforms
from common_blocks.utils import create_folds
from models.lightningclassifier import LightningClassifier

with codecs.open("config/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_tta_preds(net, images, augment=["null"]):
    with torch.no_grad():
        net.eval()
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
        folds[(folds.fold == 0)],
        config["test_inference"]["Dataset"],
        transform=get_transforms(data="valid", width=config["test_inference"]["Dataset"]["target_width"],
                                 height=config["test_inference"]["Dataset"]["target_height"]))
    loader = DataLoader(dataset, **config["test_inference"]["loader"])
    all_models = get_all_models(config["test_inference"]["models_path"])

    model_results = {"preds": [], "image_names": [], "image_label": {}}
    for fnames, images in loader:
        images = images.to(device)
        batch_preds = None
        for model in all_models:
            if batch_preds is None:
                batch_preds = get_tta_preds(model, images, augment=config["test_inference"]["TTA"])
            else:
                batch_preds += get_tta_preds(model, images, augment=config["test_inference"]["TTA"])
        model_results["image_names"].extend([i for i in fnames])
        model_results["preds"].append(batch_preds)

    model_results['preds'] = np.concatenate(model_results["preds"]).ravel() / len(all_models)
    model_results["image_label"] = list((model_results["preds"] > config["test_inference"]["threshold"]
                                         ).astype(int))

    model_results = pd.DataFrame(model_results)
    model_results['gt_label'] = folds[(folds.fold == 0)].label.reset_index(drop=True)
    class_to_id = {"correct": 1, "incorrect": 0}
    model_results['gt_label'] = model_results['gt_label'].map(class_to_id)

    print('ROC AUC', metrics.roc_auc_score(model_results['gt_label'], model_results['preds']))
    print('Precision', model_results[model_results['image_label'] == 1].gt_label.mean())
    print('Recall', metrics.recall_score(model_results['gt_label'], model_results['image_label']))
