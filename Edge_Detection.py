import cv2 as cv
import Masking as mask

testMode = 0

def getEdges(frames = mask.getFrames()):
    allPolys = {}
    for i in range(len(frames)):
        framePolys = []
        contours,_ = cv.findContours(frames[i], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for j in range(len(contours)):
            poly = []
            edges = cv.approxPolyDP(contours[j], 0.01*cv.arcLength(contours[j], True), True)
            temp = edges.ravel()
            for k in range(0, len(temp), 2):
                coordinate = (temp[k], temp[k+1])
                poly.append(coordinate)
            framePolys.append(poly)
        allPolys[i+1] = framePolys
    return allPolys

def test1(displayFrames = False):
    allQuads = {}
    frames = mask.getFrames()
    labeledFrames = [None]*len(frames)

    for i in range(len(frames)):
        frameQuads = []
        contours,_ = cv.findContours(frames[i], cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        print("Frame " + str(i + 1) + ":\n")
        labeledFrames[i] = cv.cvtColor(frames[i], cv.COLOR_GRAY2BGR)

        for j in range(len(contours)):
            quads = []
            edges = cv.approxPolyDP(contours[j], 0.01*cv.arcLength(contours[j], True), True)
            
            if len(edges) == 3: 
                print("   Shape " + str(j+1) + ": Triangle\n")
            elif len(edges) == 4: 
                print("    Shape " + str(j+1) + ": Quadrilateral\n")
            elif len(edges) == 5:
                print("    Shape " + str(j+1) + ": Pentagon\n")
            elif len(edges) == 6: 
                print("    Shape " + str(j+1) + ": Hexagon\n")
            else:
                print("   Shape " + str(j+1) + ": Pill or circle\n")
            print("        No. of non-approximated contours: " + str(len(contours[j])))
            print("        No. of approximated contours: " + str(len(edges)) + "\n")
            cv.drawContours(labeledFrames[i], edges, -1, (255,0,0), 10)

            temp = edges.ravel()
            centre = None
            sumX = 0
            sumY = 0
            for k in range(0, len(temp), 2):
                x = temp[k]
                y = temp[k+1]
                sumX += x
                sumY += y
                coordinate = (x, y)
                if len(edges) == 4: quads.append(coordinate)

                label = str(chr(ord("a") + k//2))
                print("        " + label + " : " + str(coordinate))
                cv.putText(labeledFrames[i], label, (x + 10, y + 10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
            centre = (int(sumX/(len(temp)/2)), int(sumY/(len(temp)/2)))
            print("\n        Centre = " + str(centre)+"\n")
            cv.putText(labeledFrames[i], str(j+1), centre, cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)

            if len(quads) != 0: frameQuads.append(quads)
        allQuads[i+1] = frameQuads

        print("  Therefore, quadrilaterals in this frame are: " + str(frameQuads) + "\n")
    print("Dictionary of coordinates of quadrilaterals: "+ str(allQuads))

    if displayFrames:
        cv.imshow("Original frame 1", frames[0])
        cv.imshow("Original frame 2", frames[1])
        cv.imshow("Original frame 3", frames[2])
        cv.imshow("Original frame 4", frames[3])

    cv.imshow("Labeled Frame 1", labeledFrames[0])
    cv.imshow("Labeled Frame 2", labeledFrames[1])
    cv.imshow("Labeled Frame 3", labeledFrames[2])
    cv.imshow("Labeled Frame 4", labeledFrames[3])
    if cv.waitKey(0) == ord("q"): cv.destroyAllWindows()
if testMode: test1()