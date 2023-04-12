import glob
import logging
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tqdm import tqdm
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from PIL import Image as pil_image


class Predictor:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        print("model loaded")

    def predict_directory(self, input_path: str):
        # get all files from path
        matching_files = glob.glob(os.path.join(input_path, "*.jpg"))
        matching_files_len = len(matching_files)

        predictions = list()

        if matching_files_len > 0:
            # predict each image
            for i in tqdm(range(matching_files_len), desc=f"predicting"):
                file = os.path.basename(matching_files[i])
                img = pil_image.open(matching_files[i])
                img = img.resize((200, 200), pil_image.NEAREST)
                img = np.asarray(img)
                img = img / 255.
                img = np.expand_dims(img, axis=0)

                pred = self.model.predict(img)

                pred_lab = "cat" if pred[0][0] < 0.56 else "dog"

                predictions.append({
                    "file": file,
                    "prediction": pred[0][0],
                    "prediction label": pred_lab
                })

            # save prediction
            df = pd.DataFrame(predictions)
            df.to_csv(os.path.join(input_path, "predictions.csv"), index=False)

        else:
            logging.error('No png files found.')


if __name__ == "__main__":
    model_path = 'logs/exp_000/best.hdf5'
    input_path = 'data/test'
    predictor = Predictor(model_path=model_path)
    predictor.predict_directory(input_path)
