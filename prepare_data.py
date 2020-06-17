import pandas as pd
import os

dfs = []
for folder in os.listdir("./data"):
    if "train.dir" in folder:
        print(folder, len(os.listdir("./data/" + folder)))
        for file in os.listdir("./data/" + folder):
            if file.endswith(".csv"):
                path_csv = os.path.join("./data/", folder, file)
                df = pd.read_csv(path_csv)
                df["image_filename"] = (
                    os.path.join("./data/", folder) + "/" + df["image_filename"]
                )
                dfs.append(df)
dfs = pd.concat(dfs)
dfs.to_csv("all_train.csv", index=False)
print('Unique locations', dfs['realative_coordinates'].nunique())
print('Unique locations with label = correct', dfs[dfs['label']=='correct']['realative_coordinates'].nunique())
