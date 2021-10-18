import cv2

# load some pre trained data on face frontals from opencv (haar cascade algorithm)

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in (imread = image read)
#img = cv2.imread('Robert_Downey_Jr..jpg')
img = cv2.imread('placeholder.jpg') 
# MUST convert to grayscale (only 1 instead of 3 in rgb)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # rgb is backwards in opencv

# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

print(face_coordinates)

# draw rects around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

# imshow shows image (img) with title supplied string
cv2.imshow("NibbaNaeNae's face detector", img)

# waitkey waits until a key is pressed, closes win
cv2.waitKey()

print("Terminated")

