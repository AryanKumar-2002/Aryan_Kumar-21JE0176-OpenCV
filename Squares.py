import cv2 as cv, numpy as np, math
import Masking as mask, Edge_Detection as ed

lenRatio_lower = 0.98
lenRatio_upper = 1.02

testMode = (0, 0, 0)

def getLenLine(line):
    len = ((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)**(0.5)
    return len

def getLenSquare(square):
    return getLenLine(getLines(square)[0])

def getLines(quad):
    line1 = (quad[0], quad[1])
    line2 = (quad[1], quad[2])
    line3 = (quad[2], quad[3])
    line4 = (quad[3], quad[0])

    return [line1, line2, line3, line4]

def getAngleLine(line):
    try:
        tanx = (line[0][1] - line[1][1])/(line[0][0] - line[1][0])
    except ZeroDivisionError:
        return 90

    rad = np.arctan(tanx)
    return ((rad*180/math.pi))

def getAngleSquare(square):
    return getAngleLine(getLines(square)[3])

def getCentreSquare(square):
    centreX = int((square[0][0] + square[1][0] + square[2][0] + square[3][0])/4)
    centreY = int((square[0][1] + square[1][1] + square[2][1] + square[3][1])/4)
    return (centreX, centreY)

def getSquares(allPolys = ed.getEdges()):
    allSquares = {}

    for i in range(len(allPolys)):
        framePolys = allPolys[i+1]
        frameSquares = []

        for j in range(len(framePolys)):
            poly = framePolys[j]
            if isSquare(poly): frameSquares.append(poly)

        allSquares[i+1] = frameSquares
    return allSquares

def isLenEq(line1, line2):
    ratio = getLenLine(line1)/getLenLine(line2)
    return True if (ratio >= lenRatio_lower) & (ratio <= lenRatio_upper) else False

def isSquare(poly):
    lines = getLines(poly)

    if len(poly)!=4: return False
    check1 = isLenEq(lines[0], lines[1])
    check2 = isLenEq(lines[1], lines[2])
    check3 = isLenEq(lines[2], lines[3])
    check4 = isLenEq(lines[3], lines[0])

    return True if check1 & check2 & check3 & check4 else False

def test1():
    quad = []

    for i in range(4):
        x = int(input("Enter the x coordinate of " + str(i+1) + "th edge: "))
        y = int(input("Enter the y coordinate of " + str(i+1) + "th edge: "))
        print(str(i+1) + "th coordinate : ",(x, y))
        print()
        quad.append((x, y))
    lines = getLines(quad)

    print("Length, angle of AB = " + str((getLenLine(lines[0]), int(getAngleLine(lines[0])))))
    print("Length, angle of BC = " + str((getLenLine(lines[1]), int(getAngleLine(lines[1])))))
    print("Length, angle of CD = " + str((getLenLine(lines[2]), int(getAngleLine(lines[2])))))
    print("Length, angle of DA = " + str((getLenLine(lines[3]), int(getAngleLine(lines[3])))))

    check1 = isLenEq(lines[0], lines[1])
    check2 = isLenEq(lines[1], lines[2])
    check3 = isLenEq(lines[2], lines[3])
    check4 = isLenEq(lines[3], lines[0])

    print("\nAB == BC: ", check1)
    print("BC == CD: ", check2)
    print("CD == DA: ", check3)
    print("DA == AB: ", check4)

    isSqr = isSquare(quad)
    print("Is square? : ", isSqr)
    if isSqr: print("Square inclined at ", int(getAngleLine(lines[0])), "to the horizontal.")
if testMode[0]: test1()


def test2():
    allSquares = getSquares()
    labeledFrames = [None]*len(mask.getFrames())

    for i in range(len(allSquares)):
        frameSquares = allSquares[i+1]
        labeledFrames[i] = cv.cvtColor(mask.getFrames()[i], cv.COLOR_GRAY2BGR)
        print("Frame", i+1, ":\n")

        for j in range(len(frameSquares)):
            square = frameSquares[j]
            centreX = int((square[0][0] + square[1][0] + square[2][0] + square[3][0])/4)
            centreY = int((square[0][1] + square[1][1] + square[2][1] + square[3][1])/4)
            cv.putText(labeledFrames[i], ("Frame " + str(i+1) + " square detected!!"), (centreX-150, centreY), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

            print("    Square", j+1, ":\n")
            print("        Coordinates:", str(square))
            print("        Length of a side:", int(getLenSquare(square)))
            print("        Angled at", int(getAngleSquare(square)), "degrees.\n")

    cv.imshow("Labeled Frame 1", labeledFrames[0])
    cv.imshow("Labeled Frame 2", labeledFrames[1])
    cv.imshow("Labeled Frame 3", labeledFrames[2])
    cv.imshow("Labeled Frame 4", labeledFrames[3])
    if cv.waitKey(0) == ord("q"): cv.destroyAllWindows()
if testMode[1]: test2()

def test3():
    allSquares = getSquares()
    labeledFrame = mask.getImage()

    for i in range(len(allSquares)):
        frameSquares = allSquares[i+1]
        print("Frame", i+1, ":\n")

        for j in range(len(frameSquares)):
            square = frameSquares[j]
            centreX = int((square[0][0] + square[1][0] + square[2][0] + square[3][0])/4)
            centreY = int((square[0][1] + square[1][1] + square[2][1] + square[3][1])/4)
            cv.putText(labeledFrame, ("Frame " + str(i+1) + " square detected!!"), (centreX-150, centreY), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

            print("    Square", j+1, ":\n")
            print("        Coordinates:", str(square))
            print("        Length of a side:", int(getLenSquare(square)))
            print("        Angled at", int(getAngleSquare(square)), "degrees.\n")

    cv.imshow("Labeled Frame", labeledFrame)
    if cv.waitKey(0) == ord("q"): cv.destroyAllWindows()
if testMode[2]: test3()