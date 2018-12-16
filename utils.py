"""
Author: Manksh Gupta

Stay Hungry. Stay Foolish.

"""


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def absSobelThresh(img, orient, thresh, sobelKernel=19):
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


def load_data():
    images = []
    for files in sorted_nicely(os.listdir('training_images/')):
        try:
            mypath = os.path.join('training_images', files)
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