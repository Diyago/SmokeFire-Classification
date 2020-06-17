**1. Current results**

MEAN METRIC: 0.928 std 0.085

All metrics: [0.999, 0.786, 0.937, 0.99]

![Cross Val ROC AUC](./lightning_logs/image.png)

**3. Training data**

31 unique location, only 7 with target 1. 2288 - total annotated images 

**3. To train model**

 a. Prepare data by ```prepare_data.py```
 
 b. Adjust config in `config/config_classification.yml`
 
 c. train models run ``train.py``
 
 d. Watch tensorboad logs `tensorboard --logdir ./lightning_logs/`
 
 e. All inference and deploy param specified in `deploy_model` folder
 
 d. Collect up-to-date requirements.txt call `pipreqs --force`
 
 **TODO:**
 * Collect more data 
 