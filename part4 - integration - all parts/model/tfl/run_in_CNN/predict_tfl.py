import random
from tensorflow.keras.models import load_model


def load_model(model):
    load_model(model)


def get_corner_index(row, column, img):
    row = row - random.randint(0, 81)
    column = column - random.randint(0, 81)
    return [min(max(0, row), len(img) - 81), min(max(0, column), len(img[0]) - 81)]


def crop(img, coordinates):
    return img[coordinates[0]: coordinates[0] + 81, coordinates[1]: coordinates[1] + 81]


def crop_all_images(img, candidates):
    cropped_images = []
    for c in candidates:
        cropped = crop(img, c)
        cropped_images.append(cropped)
    return cropped_images


def run_candidates_in_filter(frame, model):

    images = crop_all_images(frame.frame, frame.candidates)
    predictions = model.predict(images)

    relevant = [c for i, c in enumerate(frame.candidates) if predictions[:, 1][i] > 0.97]  # TODO by filter
    aux = [a for i, a in enumerate(frame.auxiliary) if predictions[:, 1][i] > 0.97]
    frame.traffic_lights = relevant
    frame.aux = aux

