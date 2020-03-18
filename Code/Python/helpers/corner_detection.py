import imutils
import cv2

# load the video
camera = cv2.VideoCapture(0)
last_aspectRatio = 0

# keep looping
while True:
  # grab the current frame and initialize the status text
  _, frame = camera.read()
  status = "No Targets"

  # convert the frame to grayscale, blur it, and detect edges
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (7, 7), 0)
  edged = cv2.Canny(blurred, 50, 150)
  cv2.imshow("edged",edged)

  # find contours in the edge map
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  # loop over the contours
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    # ensure that the approximated contour is "roughly" rectangular
    if len(approx) == 4:
      # compute the bounding box of the approximated contour and
      # use the bounding box to compute the aspect ratio
      (x, y, w, h) = cv2.boundingRect(approx)
      aspectRatio = w / float(h)

      # compute the solidity of the original contour
      area = cv2.contourArea(c)
      hullArea = cv2.contourArea(cv2.convexHull(c))
      solidity = area / float(hullArea)

      # compute whether or not the width and height, solidity, and
      # aspect ratio of the contour falls within appropriate bounds
      keepDims = w > 25 and h > 25
      keepSolidity = solidity > 0.9
      keepAspectRatio = aspectRatio >= 1 and aspectRatio <= 1.8

      # ensure that the contour passes all our tests
      if keepDims and keepSolidity and keepAspectRatio:
        print(x,y,w,h)
        # draw an outline around the target and update the status text
        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 4)
        last_aspectRatio = aspectRatio
        status = "Target(s) Acquired"
        # compute the center of the contour region and draw the crosshairs
        M = cv2.moments(approx)
        (cX, cY) = (int(M["m10"] / (M["m00"] + 1e-7)), int(M["m01"] / (M["m00"] + 1e-7)))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), thickness=-1, lineType=8, shift=0)

  # draw the status text on the frame
  cv2.putText(frame, status + " - Aspect Ratio: " + str('%0.3f' % last_aspectRatio), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

  # show the frame and record if a key is pressed
  cv2.imshow("Corner Detection", frame)
  key = cv2.waitKey(1) & 0xFF

  # if the 'q' key is pressed, stop the loop
  if key == ord("q"):
    break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()