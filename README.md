# Emotion recognition on collected Google images

This project uses images collected from image.google.com to train an emotion classifier.

## Current state:

It's still under development, and the test accuracy is 52%.

### Current development cycle:

1. Data collection: search & batch download on image.google.com.

2. Data processing(process_original_images.py): convert profile images to face images using OpenCV.

3. Network building: currently I use an extended LeNet-5.

4. Training: AdamOptimization or AMSGrad.

5. Predict: classify the input image that contains at least one human face.

## Example usage

```bash
# Will train the network first if a pretrained one doesn't exist.
$ python emotion_recognition.py katy_perry.jpg
```
