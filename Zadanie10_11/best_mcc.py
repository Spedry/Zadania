import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from PIL import Image as pil_image
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm


if __name__ == "__main__":
    model_path = "logs/exp_001/best.hdf5"
    model = keras.models.load_model(model_path)

    labels = pd.read_csv("data/labels.csv")
    data_root = "data/validation"

    predictions = list()

    for i, row in tqdm(labels.iterrows(), total=len(labels.index)):
        cat_name = row['image_id'].split(".")[0]

        img = pil_image.open(os.path.join(data_root, cat_name, f"{row['image_id']}.jpg"))
        img = img.resize((200, 200), pil_image.NEAREST)
        img = np.asarray(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)

        predictions.append({
            "image_id": row['image_id'],
            "prediction": pred[0][0],
        })

    predictions_df = pd.DataFrame(predictions)
    predictions_df = pd.merge(labels, predictions_df, on="image_id")

    best_thr = 0
    best_mcc = 0

    for thr in tqdm(np.arange(0., 1., 0.01)):
        this_mcc = matthews_corrcoef((predictions_df["label"]).astype("float32"), (predictions_df["prediction"] > thr).astype("float32"))
        if this_mcc > best_mcc:
            best_mcc = this_mcc
            best_thr = thr

    print(best_thr)
    print(best_mcc)

    predictions_df["prediction_label"] = (predictions_df["prediction"] > best_thr).astype("float32")
    predictions_df.to_csv("data/predictions.csv", index=False)
