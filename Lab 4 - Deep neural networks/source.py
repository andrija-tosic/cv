from enum import Enum
import numpy as np
import constants
import imutils
import cv2


class Animal(Enum):
    DOG = "DOG",
    CAT = "CAT"


def crop_scan_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ERROR = 50
    for contour in contours:
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - 1440) < ERROR and abs(h - 720) < ERROR:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                return image[y:(y + h), x:(x + w)]


def pyramid(image, scale=1.5, minSize=(30, 30)):
    yield image
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def detect_cat_dog(image, net, classes, confidence_threshold):
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))
    net.setInput(blob)
    preds = net.forward()

    idxs = np.argsort(preds[0])[::-1][:5]
    idx = idxs[0]

    if preds[0][idx] > confidence_threshold:
        if ('dog' in classes[idx]):
            return (Animal.DOG, preds[0][idx])
        elif ('cat' in classes[idx]):
            return (Animal.CAT, preds[0][idx])
    return None


if __name__ == "__main__":
    image = cv2.imread(constants.INPUT_IMAGE_PATH)
    tl_y, tl_x = 30, 110
    scan_region = image[tl_y:(tl_y + 720), tl_x:(tl_x + 1440)]
    scan_region = crop_scan_region(image)
    image = scan_region.copy()

    rows = open(constants.LABELS).read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    net = cv2.dnn.readNetFromCaffe(constants.PROTOTXT_PATH, constants.MODEL)

    for image in pyramid(image, 2, (180, 180)):
        for (x, y, tile) in sliding_window(image, stepSize=180, windowSize=(180, 180)):
            scale_w = scan_region.shape[1] / image.shape[1]
            scale_h = scan_region.shape[0] / image.shape[0]
            x_s, y_s = x, y
            w, h = int(180 * scale_w), int(180 * scale_h)
            x, y = int(x * scale_w), int(y * scale_h)

            object = detect_cat_dog(tile, net, classes, 0.4)

            if (object != None):
                animal, accuracy = object

                color = (255, 0, 0) if animal is Animal.DOG else (0, 255, 255)
                cv2.rectangle(scan_region, (x + 5, y + 5),
                              (x + h - 5, y + w - 5), color, thickness=2)
                cv2.putText(scan_region, '{} - {:.2f}'.format(str(animal.name), accuracy),
                            (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.rectangle(image, (x_s + 5, y_s + 5),
                              (x_s + 180 - 5, y + 180 - 5), color, thickness=2)

            clone = image.copy()
            cv2.rectangle(clone, (x_s, y_s),
                          (x_s + 180, y_s + 180), (0, 255, 0), 2)
            cv2.imshow("current window", clone)
            cv2.waitKey(30)

    cv2.imshow('result', scan_region)
    cv2.waitKey(0)
