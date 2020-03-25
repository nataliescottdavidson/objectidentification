import cv2

#Instantiate paths and image
faceCascadePath =  "haarcascade_frontalface_default.xml"
noseCascadePath = "haarcascade_mcs_nose.xml"
imgVirus = cv2.imread('ronies.svg', -1)
faceCascade = cv2.CascadeClassifier(faceCascadePath)
noseCascade = cv2.CascadeClassifier(noseCascadePath)

# Create the mask for the image
orig_mask = imgVirus[:, :, 3]

# Create the inverted mask for the mustache
orig_mask_inv = cv2.bitwise_not(orig_mask)

# Convert image to BGR
# and save the original image size (used later when re-sizing the image)
imgVirus = imgVirus[:, :, 0:3]
origImgHeight, origImgWidth = [int(x) for x in imgVirus.shape[:2]]


cap = cv2.VideoCapture(0)

#A function we will call later
def drawsticker():

    #Scaling image to be proportional
    imgWidth = int(nw / 1.5)
    imgHeight = int(imgWidth * origImgHeight / origImgWidth)

    # Center the mustache on the bottom of the nose
    x1 = int(nx - (imgWidth / 4))
    x2 = int(nx + nw + (imgWidth / 4))
    y1 = int(ny + nh - (imgHeight / 2))
    y2 = int(ny + nh + (imgHeight / 2))

    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # Re-calculate the width and height of the mustache image
    imgWidth = int(x2 - x1)
    imgHeight = int(y2 - y1)

    # Re-size the original image and the masks to the mustache sizes
    # calcualted above
    mustache = cv2.resize(imgVirus, (imgWidth, imgHeight), interpolation=int(cv2.INTER_AREA))
    mask = cv2.resize(orig_mask, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (imgWidth, imgHeight), interpolation=cv2.INTER_AREA)

    # take ROI for mustache from background equal to size of mustache image
    roi = roi_color[y1:y2, x1:x2]

    # roi_bg contains the original image only where the mustache is not
    # in the region that is the size of the mustache.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image of the mustache only where the mustache is
    roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    roi_color[y1:y2, x1:x2] = dst


#This is the main body of the script which runs continuously
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # i is a counter for the number of noses we see. this is problematic because we're not actually
    # persisting the identifier of the nose but it's ok for a demo
    i = 0
    # Instantiating a global variable with dummy values. Again, truly horrible programming practice
    nose1 = (0, 0)
    nose2 = (0, 0)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)

        # We save the x and y coordinates of the nose. We add x and y because nx and ny are proportional to
        # the face coordinates
        for (nx, ny, nw, nh) in nose:
            if i == 0:
                nose1 = (nx + x, ny + y)
            if i == 1:
                nose2 = (nx + x, ny + y)
            drawsticker()
            break
        i += 1

    cv2.line(frame, nose1, nose2, (0,255,0), 10)
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()