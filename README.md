# Webcam Face Pixelizer
A basic OpenCV example, which takes video stream from a webcam,
detects faces and blurs them.

```
python main.py
python main.py -s -o output_name
```


## Options from script

"-b" is blur strength. Has to be both odd and positive, as it corresponds
to kernel sizes for GaussianBlur.
"-s" saves the video
"-o" is the option for the MOV file name, default is "output"

## Dependencies
numpy==1.18.4
opencv-python==4.2.0.34
