import cv2, argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--blur_strength", type=int, default=155,
    help="Strength at which to blur, must be positive and odd")
parser.add_argument("-s", "--save_bool", action="store_true",
    help="If included, will save a video")
parser.add_argument("-o", "--output_file", type=str,
    help="Name of mov file to save", default="output")
args = parser.parse_args()

# default cascade file for frontal faces, found
# in default opencv files. Copied for convenience.
cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# Video capture object for webcam
capture = cv2.VideoCapture(0)

# Videowriter for saving videos
# Uses same dimensions as webcam
# Saves as MOV, changeable.
if args.save_bool:
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
    out = cv2.VideoWriter(
            args.output_file + '.mov', 
            fourcc, 
            20.0,
            (int(capture.get(3)), int(capture.get(4)))
            )

# Apply a (gaussian) blur to a face
# Note: The strength (which corresponds to kernel 
# size) must be both positive and odd
if args.blur_strength % 2 == 0:
    args.blur_strength += 1
    print("Blur Strength changed to  ", args.blur_strength)

def blur(face, strength):
    noisy_face = cv2.GaussianBlur(face, (strength, strength), 0)
    return noisy_face

# Main loop, exits on "q"
while(True):
    # Open Frame
    ret, frame = capture.read()
   
    # Find faces from a grayscale copy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Blur face
    for (x,y,w,h) in faces:
        curr_face = frame[y:y+h, x:x+h]
        blurred_face = blur(curr_face, strength = args.blur_strength)
        frame[y:y+h, x:x+h] = blurred_face

    # show vid
    cv2.imshow('Video', frame)
    if args.save_bool: out.write(frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release capture and outfile
capture.release()
if args.save_bool: out.release()

# Destroy remaining windows
cv2.destroyAllWindows()

