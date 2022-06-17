import cv2 as cv, numpy as np, math
import Masking as mask, Squares as sq, Aruco as aruco

image = mask.getImage()

temp = sq.getSquares()
squares = [temp[1][0], temp[2][0], temp[3][0]]
sizeOfSquares = [sq.getLenSquare(squares[0]), sq.getLenSquare(squares[1]), sq.getLenSquare(squares[2])]
angleOfSquares = [sq.getAngleSquare(squares[0]), sq.getAngleSquare(squares[1]), sq.getAngleSquare(squares[2])]

temp = aruco.getMarkerSources()
markers = [temp[0], temp[1], temp[2]]
temp,_ = aruco.getMarkersInfo(markers)
cofMarkers = [temp[1][0], temp[2][0], temp[3][0]]

dim1 = image.shape[:2][0]
dim2 = image.shape[:2][1]
stickerboard = np.zeros((dim1, dim2), np.uint8)
stickerboard.fill(255)

window = image.copy()

for i in range(len(squares)):

    angleMarker = sq.getAngleSquare(cofMarkers[i])
    angleSquare = angleOfSquares[i]
    colored = markers[i].copy()

    lower_black = np.array([0,0,0], dtype = "uint16")
    upper_black = np.array([220,220,220], dtype = "uint16")
    for j in range (lower_black[0], upper_black[0]): colored[np.where((colored==[j,j,j]).all(axis=2))] = [235, 64, 52]

    padded = aruco.addPadding(colored)
    centre = aruco.getCentreImage (padded)
    rotated = aruco.rotate(padded, -angleSquare + angleMarker, centre)
   
    for j in range (lower_black[0], upper_black[0]): rotated[np.where((rotated==[j,j,j]).all(axis=2))] = [255, 255, 255]
    rotated[np.where((rotated==[235, 64, 52]).all(axis=2))] = [0, 0, 0]
    temp,_ = aruco.getMarkersInfo([rotated])
    rotatedMarker = temp[1][0]

    cropped = aruco.crop(rotated, rotatedMarker)
    
    x1, x2, y1, y2 = squares[i][0][0], squares[i][0][0], squares[i][0][1], squares[i][0][1]
    for k in range (4):
        if squares[i][k][0] < x1: x1 = squares[i][k][0]
        if squares[i][k][0] > x2: x2 = squares[i][k][0]
        if squares[i][k][1] < y1: y1 = squares[i][k][1]
        if squares[i][k][1] > y2: y2 = squares[i][k][1]

    width = x2 - x1
    height = y2 - y1
    resized = cv.resize(cropped, (width,height))

    sticker = resized.copy()
    stickerboard[y1 : y2, x1 : x2]= cv.cvtColor(sticker, cv.COLOR_BGR2GRAY)

    points = np.array([list(squares[i][0]),list(squares[i][1]),list(squares[i][2]),list(squares[i][3])], np.int32).reshape(-1,1,2)
    rectangle = cv.minAreaRect(points)
    box = cv.boxPoints(rectangle)
    box = np.int0(box)
    cv.drawContours(window,[box],0,(255,255,255),-1)

result = cv.bitwise_and(window, window, mask = stickerboard)

cv.imshow("Final.jpg", result)
if cv.waitKey(0) == ord("q"): cv.destroyAllWindows()
#cv.imwrite("Final.jpg", result) 