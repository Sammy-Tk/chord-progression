from chords_prog_proj.ml_logic.params import LOCAL_REGISTRY_PATH

import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow.keras import Model, models


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        output_dir = os.path.join(LOCAL_REGISTRY_PATH, "params")
        params_path = os.path.join(output_dir, f"{timestamp}.pickle")
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        with open(params_path, "wb") as file:
            pickle.dump(params, file)
        print(f"✅ Params saved at {params_path}")

    # save metrics
    if metrics is not None:
        output_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        metrics_path = os.path.join(output_dir, f"{timestamp}.pickle")
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
        print(f"✅ Metrics saved at {metrics_path}")

    # save model
    if model is not None:
        model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
        os.makedirs(model_dir, exist_ok=True)  # create directory if it doesn't exist
        model_path = os.path.join(model_dir, f"{timestamp}.keras")
        model.save(model_path)
        print(f"✅ Model saved at {model_path}")

        print("\n✅ data saved locally")

    return None


def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model


def get_model_version(stage="Production"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    # model version not handled

    return None
