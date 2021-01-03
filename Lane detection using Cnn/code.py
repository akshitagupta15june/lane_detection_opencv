import numpy as np
import cv2
from scipy.misc import imresize
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import model_from_json
import pickle
import matplotlib.image as pimg
json_file = open('CNN_model.json', 'r')
json_model = json_file.read()
json_file.close()
model = model_from_json(json_model)
model.load_weights('CNN_model.h5')
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []
def road_lines_image(imageIn):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    #crop to 720x1280, img[y: y + h, x: x + w], 300:940
    image = imageIn[230:950, 0:1280]
    image = imresize(image, (640, 1280, 3))
    
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, image.shape)
    
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result
lanes = Lanes()
def road_lines_fc(imageIn):
    img_det = road_lines_image(imageIn)

    cv2.imshow('detection',img_det )  

    cv2.waitKey(0)
for xx in range(0, 1 ): 
    imageIn = pimg.imread("./1.png")
    road_lines_fc(imageIn)
    imageIn = pimg.imread("./2.png")
    road_lines_fc(imageIn)
cv2.destroyAllWindows()