This is a video processing frame-work in which we are using motion estimation and DCT compression to encode and decode the live video stream.

# in encoder.py we have
* taken 25 frames from live video stream using webcam
* applied YCbCr conversion, chroma subsampling, pyramidal filtering and DCT compression on every I-Frame
* calculated the motion vectors on P-Frame
* stored DCT coeffecients for I-Frames and motion vectors for P-Frames

in decoder.py we have
* applied inverse motion estimation on P-Frames
* applied inverse DCT,inserted zeros, filtering on I frames
* applied YCbCr to RGB conversion

# Usage
python encoder.py to encode the frames                 
python decoder.py to decode the frames