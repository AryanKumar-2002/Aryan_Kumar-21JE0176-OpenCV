import cv2 as cv, numpy as np

__image = cv.resize(cv.imread("assets\CVtask.jpg"), (0,0),  fx = 0.3, fy = 0.3)
__colours = {

"Orange" : np.array([[10,100,20], [25,255,255]], np.uint8),
"Green" : np.array([[25,52,72], [102,255,255]], np.uint8),
"Black" : np.array([[0,0,0], [180,255,150]], np.uint8),
"Pink" : np.array([[0,249,240], [255,255,255]], np.uint8)

}

__kernel = np.ones((2,2), np.uint0)

testMode = 0

def setImage(image= __image):
    global __image
    __image = image

def setColours(colours= __colours):
    global __colours
    __colours = colours

def setKernel(kernel):
    global __kernel
    __kernel = kernel

def getImage():
    global __image
    return __image.copy()

def getColours():
    global __colours
    return __colours.copy()

def getKernel():
    global __kernel
    return __kernel.copy()

def __getKeys(colours):
    keys = {}
    for i in range(len(colours.keys())):
        keys[i+1] = list(colours.keys())[i]
    return keys

def getKeys():
    return __getKeys(__colours)

def showImage(text = "Original image"):
    cv.imshow(text, getImage())

def getFrames(image = getImage()):
    cvtImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    colours = getColours()
    kernel = getKernel()
    badFrames = []
    frames = []

    for bounds in colours: badFrames.append(cv.inRange(cvtImage, colours[bounds][0], colours[bounds][1]))
    for badFrame in badFrames:
        frames.append(cv.morphologyEx(cv.morphologyEx(badFrame,cv.MORPH_OPEN,kernel), cv.MORPH_CLOSE,kernel))
    return frames

def test1():
    print("Keys:", getKeys())
    frames = getFrames()
    showImage()

    cv.imshow("Orange", frames[0])
    cv.imshow("Green", frames[1])
    cv.imshow("Black", frames[2])
    cv.imshow("Pink-Peach", frames[3])
    if cv.waitKey(0) == ord("q"): cv.destroyAllWindows()
if testMode: test1()