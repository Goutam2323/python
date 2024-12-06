import cv2  # Importing OpenCV


# Loading the Haar Cascades for face and cat face detection
human_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')


# Initialize camera
cam = cv2.VideoCapture(0)


while True:
    # Read the camera frame
    _, img = cam.read()


    # Convert the frame to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Detect human faces in the image
    human_faces = human_cascade.detectMultiScale(grayImg, 1.3, 4)


    # Detect cat faces in the image
    cat_faces = cat_cascade.detectMultiScale(grayImg, 1.3, 4)


    # Draw rectangles around detected human faces
    for (x, y, w, h) in human_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box for humans


    # Draw rectangles around detected cat faces
    for (x, y, w, h) in cat_faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue box for cats


    # Display the frame with detected human and cat faces
    cv2.imshow("Human and Cat Face Detection", img)


    # Exit when 'Escape' key is pressed
    key = cv2.waitKey(10)
    if key == 27:
        break


# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()


