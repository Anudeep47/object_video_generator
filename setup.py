import sys
import os
from cv2 import VideoWriter, VideoWriter_fourcc, VideoCapture
import object_detection as od

def make_object_videos(video_path):
    fourcc = VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = 2
    size = None
    ext = '.mp4'
    video = VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = video.read()
        if size is None:
            size = image.shape[1], image.shape[0]
            video_dict = {
                'person': VideoWriter('people_video'+ext, fourcc, float(fps), size), 
                'bicycle': VideoWriter('bicycle_video'+ext, fourcc, float(fps), size), 
                'car': VideoWriter('car_video'+ext, fourcc, float(fps), size), 
                'motorcycle': VideoWriter('motorcycle_video'+ext, fourcc, float(fps), size), 
                'bus': VideoWriter('bus_video'+ext, fourcc, float(fps), size), 
                'truck': VideoWriter('truck_video'+ext, fourcc, float(fps), size)
            }
        objects = od.detect_objects_in_frame(image)
        print(objects)
        for obj in objects:
            video_dict[obj].write(image)
        count += 1
        # Uncomment below code to only read upto 30 frames in video
        # if count == 30:  
        #     break

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = os.path.abspath(sys.argv[1])
        print(path)
        make_object_videos(path)
    else:
        print('\nUSAGE: python3 setup.py <relative-path-to-video-file>\n')
        sys.exit()