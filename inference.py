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


"""
Config values used throughout the code.
"""

from os import PathLike

import yaml
from pydantic import BaseModel, computed_field, model_validator


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
    """
    Image width which is fed to the model.
    """

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


def load_config_from_yaml(yaml_file_path: str | PathLike[str]) -> PlateOCRConfig:
    """Read and parse a yaml containing the Plate OCR config."""
    with open(yaml_file_path, encoding="utf-8") as f_in:
        yaml_content = yaml.safe_load(f_in)
    config = PlateOCRConfig(**yaml_content)
    return config


def one_hot_plate(plate: str, alphabet: str) -> list[list[int]]:
    return [[0 if char != letter else 1 for char in alphabet] for letter in plate]


def target_transform(
    plate_text: str,
    max_plate_slots: int,
    alphabet: str,
    pad_char: str,
) -> npt.NDArray[np.uint8]:
    # Pad the plates which length is smaller than 'max_plate_slots'
    plate_text = plate_text.ljust(max_plate_slots, pad_char)
    # Generate numpy arrays with one-hot encoding of plates
    encoded_plate = np.array(one_hot_plate(plate_text, alphabet=alphabet), dtype=np.uint8)
    return encoded_plate


def read_plate_image(image_path: str, img_height: int, img_width: int) -> npt.NDArray:
    """
    Read and resize a license plate image.

    :param image_path: The path to the license plate image.
    :param img_height: The desired height of the resized image.
    :param img_width: The desired width of the resized image.
    :return: The resized license plate image as a NumPy array.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
    img = np.expand_dims(img, -1)
    return img


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


def load_images_from_folder(
    img_dir: pathlib.Path,
    width: int,
    height: int,
    shuffle: bool = False,
    limit: int | None = None,
) -> list[npt.NDArray]:
    """
    Return all images read from a directory. This uses the same read function used during training.
    """
    image_paths = sorted(
        str(f.resolve()) for f in img_dir.iterdir() if f.is_file() and f.suffix in IMG_EXTENSIONS
    )
    if limit:
        image_paths = image_paths[:limit]
    if shuffle:
        random.shuffle(image_paths)
    images = [read_plate_image(i, img_height=height, img_width=width) for i in image_paths]
    return images


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


def low_confidence_positions(probs, thresh=0.3) -> npt.NDArray:
    """Returns indices of elements in `probs` less than `thresh`, indicating low confidence."""
    return np.where(np.array(probs) < thresh)[0]


def display_predictions(
    image: npt.NDArray,
    plate: str,
    probs: npt.NDArray,
    low_conf_thresh: float,
) -> None:
    """
    Display plate and corresponding prediction.
    """
    plate_str = "".join(plate)
    logging.info("Plate: %s", plate_str)
    logging.info("Confidence: %s", probs)
    image_to_show = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    # Converting to BGR for color text
    image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2RGB)
    # Average probabilities
    avg_prob = np.mean(probs) * 100
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 0),
        lineType=1,
        thickness=6,
    )
    cv2.putText(
        image_to_show,
        f"{plate_str}  {avg_prob:.{2}f}%",
        org=(5, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(255, 255, 255),
        lineType=1,
        thickness=2,
    )
    # Display character with low confidence
    low_conf_chars = "Low conf. on: " + " ".join(
        [plate[i] for i in low_confidence_positions(probs, thresh=low_conf_thresh)]
    )
    cv2.putText(
        image_to_show,
        low_conf_chars,
        org=(5, 200),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=(0, 0, 220),
        lineType=1,
        thickness=2,
    )
    cv2.imshow("plates", image_to_show)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        return

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
    images = load_images_from_folder(
        img_dir, width=config.img_width, height=config.img_height
    )
    for image in images:
        x = np.expand_dims(image, 0)
        prediction = model(x, training=False)
        prediction = keras.ops.stop_gradient(prediction).numpy()
        plate, probs = postprocess_model_output(
            prediction=prediction,
            alphabet=config.alphabet,
            max_plate_slots=config.max_plate_slots,
            vocab_size=config.vocabulary_size,
        )
        print(plate)

    cv2.destroyAllWindows()
