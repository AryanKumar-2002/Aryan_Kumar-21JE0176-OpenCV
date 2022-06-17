import cv2 as cv, numpy as np, math
import Masking as mask, Squares as sq, Aruco as aruco

#gets the image containing the various shapes. Note the original image has been resized to a more convenient size. 
#The output for the original image will be similar, and this can be verified easily, 
#by removing the resize function I used on the original image in Masking.py file.
image = mask.getImage()

#gets the coordinates of squares in the image. The temp variable in this code can be safely ignored
temp = sq.getSquares()
squares = [temp[1][0], temp[2][0], temp[3][0]]

#gets the ArUco marker images, and stores them in "markers" variable. The cofMarkers then stores their coordinates.
#Note the markers have been arranged according to their ids. The ids can easily be found using toolos in Aruco.py.
markers =  aruco.getMarkerSources()
temp,_ = aruco.getMarkersInfo(markers)
cofMarkers = [temp[1][0], temp[2][0], temp[3][0]]

#stickerboard is where we will paste the markers initially. The stickerboard is exactly the same size as image, 
#and the markers pasted on it exactly corresponds to the location of squares on the image.
stickerboard = np.zeros((image.shape[:2][0], image.shape[:2][1]), np.uint8)
stickerboard.fill(255)

#window will be a copy of the image, except for that the squares in image will have been whitened.
window = image.copy()

#Our approach will be to build up stickerboard and window as above. 
#Then, we will add the two. The markers of the stickerboard will "shine" through the whitened squares, "windows" of the window image.

for i in range(3):

    #centre stores the coordinates of the centre of the marker, while angleMarker stores its orientation. 
    #Using the info, the marker is rotated about its centre to make its sides horizontal/vertical.
    centre = sq.getCentreSquare(cofMarkers[i])
    angleMarker = sq.getAngleSquare(cofMarkers[i])
    rotated = aruco.rotate(markers[i], angleMarker, centre)

    #rotatedMarker saves the coordinates of the rotated marker
    temp,_ = aruco.getMarkersInfo([rotated])
    rotatedMarker = temp[1][0]
    cropped = aruco.crop(rotated, rotatedMarker)

    #the marker is then resized to the size of the square
    resized = cv.resize(cropped, (int(sq.getLenSquare(squares[i])), int(sq.getLenSquare(squares[i]))))

    #due to the rotate operation we performed on the marker above, the image's pixel values on the boundaries have changed.
    #the image is now no longer pure black and white (can be easily verified by printing the image to a text file).
    #this is a matter of concern, since we shall perform some pixel manipulatons on black later.
    #Adding to the problem, I used rotation twice in the program in pursuit to keep its code simple.
    #Now nobody likes blocks of pixels as an image in this era of UHD displas on our smartphones.
    #Thus, a sharpenening algorithm was needed. Returns pure black and white image.
    lower_black = np.array([0,0,0], np.uint8)
    upper_black = np.array([100,100,100], np.uint8)
    sharpened = cv.bitwise_not(cv.inRange(resized, lower_black, upper_black))
    sharpened = cv.cvtColor(sharpened, cv.COLOR_GRAY2RGB)

    #At this point, we are faced with another problem. Rotation will introduce regions of black in the image.
    #We will need to convert those blacks to white. In order to differentiate those blacks from the blacks of the ArUco,
    #we will convert the blacks in ArUco to blue. (235, 64, 52) is the BGR vale of a shade of blue, my fav colour ;)
    coloured = sharpened.copy()
    coloured[np.where((coloured==[0,0,0]).all(axis=2))] = [235, 64, 52]

    #Before rerotating, let us add some padding to ensure the marker does not get cut off.
    angleSquare = sq.getAngleSquare(squares[i])
    padded = aruco.addPadding(coloured, (90 - angleSquare))

    #Now we rerotate the marker to match the orientation of the squares.
    centre2 = aruco.getCentreImage(padded)
    rerotated = aruco.rotate(padded, -angleSquare, centre2)

    #Now we turn the blacks introduced due to rotation to white.
    corrected = rerotated.copy()
    lower_black = np.array([0,0,0], np.uint8)
    upper_black = np.array([220,220,220], np.uint8)
    for j in range (lower_black[0], upper_black[0]): corrected[np.where((corrected==[j,j,j]).all(axis=2))] = [255, 255, 255]

    #Next up, the markers are pasted on the stickerboard, after the blue colour in them has been changed to white.
    #They are pasted on the exact same location in stcikerboard, as the position of squares in the image.
    x_offset, y_offset = squares[i][0][0], squares[i][0][1]
    for k in range (4):
        if squares[i][k][0] < x_offset: x_offset = squares[i][k][0]
        if squares[i][k][1] < y_offset: y_offset = squares[i][k][1]
    x_end = x_offset + corrected.shape[1]
    y_end = y_offset + corrected.shape[0]

    sticker = corrected.copy()
    sticker[np.where((sticker==[235, 64, 52]).all(axis=2))] = [0,0,0]

    stickerboard[y_offset : y_end, x_offset : x_end]= cv.cvtColor(sticker, cv.COLOR_BGR2GRAY)

    #Here, we whiten the squares in window, by making white squares over them.
    points = np.array([list(squares[i][0]), list(squares[i][1]), list(squares[i][2]), list(squares[i][3])], np.int32).reshape(-1,1,2)
    rectangle = cv.minAreaRect(points)
    box = cv.boxPoints(rectangle)
    box = np.int0(box)
    cv.drawContours(window,[box],0,(255,255,255),-1)

#Finally, the moment of truth. We add the two images.
result = cv.bitwise_and(window, window, mask = stickerboard)
cv.imwrite("Final.jpg", result) 