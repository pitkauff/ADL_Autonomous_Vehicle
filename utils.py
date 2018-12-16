
import os
from PIL import Image
import numpy as np
import cv2
import re
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K

def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def absSobelThresh(img, orient, thresh, sobelKernel=19):

    """

    :param img: image
    :param orient: orientation
    :param thresh: thresholds
    :param sobelKernel: size of the filter
    :return: binary image output
    """
    threshMin = thresh[0]
    threshMax = thresh[1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobelOp = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobelKernel)
    else:
        sobelOp = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobelKernel)
    absSobel = np.absolute(sobelOp)
    scaledSobel = np.uint8(255 * absSobel / np.max(absSobel))
    sxbinary = np.zeros_like(scaledSobel)
    sxbinary[(scaledSobel > threshMin) & (scaledSobel < threshMax)] = 1
    binaryOutput = sxbinary

    return binaryOutput


def combinedThreshBinaryImg(img, threshX, threshY, threshColorS, threshColorU, threshColorR):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float)
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    R = rgb[:, :, 0]
    U = yuv[:, :, 1]
    sobelX = absSobelThresh(img, orient='x', thresh=(threshX[0], threshX[1]))
    sobelY = absSobelThresh(img, orient='y', thresh=(threshY[0], threshY[1]))
    sBinary = np.zeros_like(S)
    sBinary[(S >= threshColorS[0]) & (S <= threshColorS[1])] = 1
    rBinary = np.zeros_like(R)
    rBinary[(R >= threshColorR[0]) & (R <= threshColorR[1])] = 1
    uBinary = np.zeros_like(U)
    uBinary[(U >= threshColorU[0]) & (U <= threshColorU[1])] = 1
    colorBinary = np.dstack((rBinary, ((sobelX == 1) & (sobelY == 1)), uBinary))
    combinedBinary = np.zeros_like(sBinary)
    combinedBinary[(rBinary == 1) | (uBinary == 1) | ((sobelX == 1) & (sobelY == 1))] = 1

    return combinedBinary


def heatmap(images, model):

    """
    This function creates a heatmap of the convolution layers and helps to visualise what the model is learning.
    :param images: Image
    :param model: model that is trained
    :return: heatmap of filter
    """
    final_images = []
    k = 1
    for i in range(len(images)):
        fig = plt.figure(figsize=(20, 6 * len(images)))
        im = np.expand_dims(images[i], axis=0)
        preds = model.predict(im)
        output = model.get_layer("dense_2").output[:, np.argmax(preds)]
        last_conv_layer = model.get_layer('conv2d_2')
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        grads_value, conv_layer_output_value = iterate([im])
        for j in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, j] *= grads_value[j]
        heatmap_1 = np.mean(conv_layer_output_value, axis=- 1)
        heatmap_1 = np.maximum(heatmap_1, 0)
        heatmap_1 /= np.max(heatmap_1)
        fig.add_subplot(len(images), 3, k)
        plt.imshow(heatmap_1)

        heatmap_2 = cv2.resize(heatmap_1, (images[i].shape[1], images[i].shape[0]))
        heatmap_2 = np.uint8(255 * heatmap_2)
        heatmap_2 = cv2.applyColorMap(heatmap_2, cv2.COLORMAP_JET)
        fig.add_subplot(len(images), 3, k + 1)
        plt.imshow(heatmap_2)

        superimposed_img = heatmap_2 * 0.7 + images[i] * 255
        superimposed_img = np.array(superimposed_img).astype('uint8')
        fig.add_subplot(len(images), 3, k + 2)
        plt.imshow(superimposed_img)
        final_images.append(superimposed_img)
        k += 3

    return np.array(final_images)


def load_data(data_directory):
    images = []
    for files in sorted_nicely(os.listdir(data_directory)):
        try:
            mypath = os.path.join(data_directory, files)
            img = Image.open(mypath, mode='r')
            img = img.resize((128, 128))
            arr = np.array(img).astype('uint8')
            arr = combinedThreshBinaryImg(arr, threshX=(1, 255),
                                          threshY=(50, 255),
                                          threshColorS=(1, 255),
                                          threshColorU=(250, 250),
                                          threshColorR=(230, 255))
            arr = arr[60:, ]
            images.append(arr)
            img.close()
        except:
            pass

    return images


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes, COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detection(image, weights, cfg, classes_file):
    weights = weights
    config = cfg
    image = cv2.imread(image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights, config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h), classes,
                        COLORS)

    return image, boxes



