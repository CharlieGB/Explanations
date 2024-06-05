import os
import cv2

directories = ['2020_11_25_15_03_34_921934']

for directory in directories:
    image_folder = os.getcwd() + '/Results/' + directory + '/'
    speed = 6

    # Policy video on test trial
    video_name = 'Policy.avi'

    num_images = len([img for img in os.listdir(image_folder)if 'Test' in img])
    images = ['Test' + str(i) + '.png' for i in range(num_images)]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(image_folder + video_name, 0, speed / 2, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


    # Eta heat map during training
    video_name = 'EtaHeatMap.avi'

    num_images = len([img for img in os.listdir(image_folder)if 'EtaHeatMap' in img])
    images = ['EtaHeatMap' + str(i) + '.png' for i in range(num_images)]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(image_folder + video_name, 0, speed, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

    # Value function during training
    video_name = 'ValueFunction.avi'

    num_images = len([img for img in os.listdir(image_folder)if 'ValueFunction' in img])
    images = ['ValueFunction' + str(i) + '.png' for i in range(num_images)]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(image_folder + video_name, 0, speed, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
