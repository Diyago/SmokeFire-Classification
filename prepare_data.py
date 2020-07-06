import os

import pandas as pd


def prepate_train_csv(path):
    dfs = []
    for folder in os.listdir(path):
        if "train.dir" in folder:
            print(folder, len(os.listdir(path + folder)))
            for file in os.listdir(path + folder):
                if file.endswith(".csv"):
                    path_csv = os.path.join(path, folder, file)
                    df = pd.read_csv(path_csv)
                    df["image_filename"] = (
                            os.path.join(path, folder) + "/" + df["image_filename"]
                    )
                    dfs.append(df)
    dfs = pd.concat(dfs)
    dfs.to_csv("all_train.csv", index=False)
    print('Unique locations', dfs['realative_coordinates'].nunique())
    print('Unique locations with label = correct', dfs[dfs['label'] == 'correct']['realative_coordinates'].nunique())


if __name__ == '__main__':
    prepate_train_csv('./data/smoke_train/')
