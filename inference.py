from tensorflow import keras
from keras import losses, metrics, ops
import logging
import pathlib
import random
import cv2
import keras
import numpy as np
import numpy.typing as npt
import pathlib
import logging
import pathlib
import random
from os import PathLike
import yaml
from pydantic import BaseModel, computed_field, model_validator


"""
Custom metrics and loss functions.
"""
def cat_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Categorical accuracy metric.
    """

    def cat_acc(y_true, y_pred):
        """
        This is simply the CategoricalAccuracy for multi-class label problems. Example if the
        correct label is ABC123 and ABC133 is predicted, it will not give a precision of 0% like
        plate_acc (not completely classified correctly), but 83.3% (5/6).
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        return ops.mean(metrics.categorical_accuracy(y_true, y_pred))

    return cat_acc


def plate_acc_metric(max_plate_slots: int, vocabulary_size: int):
    """
    Plate accuracy metric.
    """

    def plate_acc(y_true, y_pred):
        """
        Compute how many plates were correctly classified. For a single plate, if ground truth is
        'ABC 123', and the prediction is 'ABC 123', then this would give a score of 1. If the
        prediction was ABD 123, it would score 0.
        """
        y_true = ops.reshape(y_true, newshape=(-1, max_plate_slots, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, max_plate_slots, vocabulary_size))
        et = ops.equal(ops.argmax(y_true, axis=-1), ops.argmax(y_pred, axis=-1))
        return ops.mean(ops.cast(ops.all(et, axis=-1, keepdims=False), dtype="float32"))

    return plate_acc


def top_3_k_metric(vocabulary_size: int):
    """
    Top 3 K categorical accuracy metric.
    """

    def top_3_k(y_true, y_pred):
        """
        Calculates how often the true character is found in the 3 predictions with the highest
        probability.
        """
        # Reshape into 2-d
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        return ops.mean(metrics.top_k_categorical_accuracy(y_true, y_pred, k=3))

    return top_3_k


# Custom loss
def cce_loss(vocabulary_size: int, label_smoothing: float = 0.2):
    """
    Categorical cross-entropy loss.
    """

    def cce(y_true, y_pred):
        """
        Computes the categorical cross-entropy loss.
        """
        y_true = ops.reshape(y_true, newshape=(-1, vocabulary_size))
        y_pred = ops.reshape(y_pred, newshape=(-1, vocabulary_size))
        return ops.mean(
            losses.categorical_crossentropy(
                y_true, y_pred, from_logits=False, label_smoothing=label_smoothing
            )
        )

    return cce


class PlateOCRConfig(BaseModel, extra="forbid", frozen=True):
    """
    Model License Plate OCR config.
    """
    max_plate_slots: int
    """
    Max number of plate slots supported. This represents the number of model classification heads.
    """
    alphabet: str
    """
    All the possible character set for the model output.
    """
    pad_char: str
    """
    Padding character for plates which length is smaller than MAX_PLATE_SLOTS.
    """
    img_height: int
    """
    Image height which is fed to the model.
    """
    img_width: int


    @computed_field  # type: ignore[misc]
    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet)

    @model_validator(mode="after")
    def check_pad_in_alphabet(self) -> "PlateOCRConfig":
        if self.pad_char not in self.alphabet:
            raise ValueError("Pad character must be present in model alphabet.")
        return self


def load_config_from_yaml(yaml_file_path: str | PathLike[str]) -> PlateOCRConfig:
    """Read and parse a yaml containing the Plate OCR config."""
    with open(yaml_file_path, encoding="utf-8") as f_in:
        yaml_content = yaml.safe_load(f_in)
    config = PlateOCRConfig(**yaml_content)
    return config

def one_hot_plate(plate: str, alphabet: str) -> list[list[int]]:
    return [[0 if char != letter else 1 for char in alphabet] for letter in plate]


def load_keras_model(
    model_path: str | pathlib.Path,
    vocab_size: int,
    max_plate_slots: int,
) -> keras.Model:
    """
    Utility helper function to load the keras OCR model.
    """
    custom_objects = {
        "cce": cce_loss(vocabulary_size=vocab_size),
        "cat_acc": cat_acc_metric(max_plate_slots=max_plate_slots, vocabulary_size=vocab_size),
        "plate_acc": plate_acc_metric(max_plate_slots=max_plate_slots, vocabulary_size=vocab_size),
        "top_3_k": top_3_k_metric(vocabulary_size=vocab_size),
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    return model


IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
"""Valid image extensions for the scope of this script."""


def postprocess_model_output(
    prediction: npt.NDArray,
    alphabet: str,
    max_plate_slots: int,
    vocab_size: int,
) -> tuple[str, npt.NDArray]:
    """
    Return plate text and confidence scores from raw model output.
    """
    prediction = prediction.reshape((max_plate_slots, vocab_size))
    probs = np.max(prediction, axis=-1)
    prediction = np.argmax(prediction, axis=-1)
    plate = "".join([alphabet[x] for x in prediction])
    return plate, probs

if __name__ == "__main__":
    model_path = r"C:\\Users\\Acer\\Desktop\\cnn_ocr-epoch_728-acc_0.922.keras"
    config_file = r"C:\\Users\\Acer\\Desktop\\config.yaml"

    from pathlib import Path
    img_path = r"C:\\Users\\Acer\\Desktop\\img"
    img_dir = Path(img_path)

    config = load_config_from_yaml(config_file)
    model = load_keras_model(
        model_path, vocab_size=config.vocabulary_size, max_plate_slots=config.max_plate_slots
    )

    cv2.destroyAllWindows()
