import codecs
import os
import sys
import warnings
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from albumentations import Compose, Normalize
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, HTTPException
from loguru import logger
from poyo import parse_string
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from models.lightningclassifier import LightningClassifier

warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_transforms(width, height):
    assert width % 32 == 0
    assert height % 32 == 0

    return Compose(
        [transforms.Resize(width, height, always_apply=True),
         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ToTensorV2(),
         ]
    )


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
with codecs.open("models/config_classification.yml", encoding="utf-8") as ymlfile:
    config_yaml = ymlfile.read()
    config = parse_string(config_yaml)

if __name__ == 'main':
    app = FastAPI()
    logger.add("file_1.log", backtrace=True, rotation="100 MB")  # Automatically rotate too big file
    logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
    all_models = get_all_models(config["test_inference"]["models_path"])


    @app.get("/ping")
    def ping():
        logger.info('ping POST request performed')
        return {"message": "Server is UP!"}


    @app.post("/predict_image/")
    @logger.catch
    def make_inference(file: bytes = File(...)):
        try:
            pil_image = np.array(Image.open(BytesIO(file)))

        except:
            raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file"
            )
        logger.info('predict_image POST request performed. img shape {}'.format(pil_image.shape))
        augmented = get_transforms(config['test_inference']['Dataset']['target_width'],
                                   config['test_inference']['Dataset']['target_height'])(image=pil_image)
        pil_image = torch.from_numpy(np.expand_dims(augmented['image'], axis=0)).to(device)
        logger.info('predict_image POST augmented img shape {}, type {}'.format(pil_image.shape, pil_image.dtype))
        batch_preds = None
        for model in all_models:
            if batch_preds is None:
                batch_preds = get_tta_preds(model, pil_image, augment=config["test_inference"]["TTA"])
            else:
                batch_preds += get_tta_preds(
                    model, pil_image, augment=config["test_inference"]["TTA"]
                )
        model_results = {"preds": [], "image_label": {}, "model_type": config["test_inference"]["model_type"]}

        model_results["preds"].append(batch_preds / len(all_models))
        model_results['preds'] = np.concatenate(model_results["preds"]).ravel()
        model_results["image_label"] = (model_results["preds"] > config["test_inference"]["threshold"]
                                        ).astype(int).tolist()
        model_results['preds'] = model_results['preds'].tolist()
        logger.info('Results {}'.format(model_results))

        return model_results
