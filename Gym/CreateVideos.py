import os
import cv2

root_dir = os.getcwd() + '/Results/Explanations/'
directories = os.listdir(root_dir)

explanation_lengths = [5, 3]
explanation_thresholds = [.5, .75, .9]

for directory in directories:
    image_folder = root_dir + directory + '/'
    speed = 6

    # Policy video on test trial
    video_name = 'AllPolicy.avi'
    num_images = len([img for img in os.listdir(image_folder)if 'AllTest' in img])
    images = ['AllTest' + str(i) + '.png' for i in range(num_images)]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(image_folder + video_name, 0, speed / 2, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()



    for length in explanation_lengths:
        # Policy video on test trial
        video_name = 'Length' + str(length) + 'Policy.avi'
        num_images = len([img for img in os.listdir(image_folder) if 'Length' + str(length) in img])
        images = ['Length' + str(length) + 'Test' + str(i) + '.png' for i in range(num_images)]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(image_folder + video_name, 0, speed / 2, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()



    for threshold in explanation_thresholds:
        # Policy video on test trial
        video_name = 'Treshold' + str(threshold) + 'Policy.avi'
        num_images = len([img for img in os.listdir(image_folder) if 'Threshold' + str(threshold) in img])
        images = ['Threshold' + str(threshold) + 'Test' + str(i) + '.png' for i in range(num_images)]

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(image_folder + video_name, 0, speed / 2, (width, height))
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

    # # Value function during training
    # video_name = 'ValueFunction.avi'
    #
    # num_images = len([img for img in os.listdir(image_folder)if 'ValueFunction' in img])
    # images = ['ValueFunction' + str(i) + '.png' for i in range(num_images)]
    #
    # frame = cv2.imread(os.path.join(image_folder, images[0]))
    # height, width, layers = frame.shape
    #
    # video = cv2.VideoWriter(image_folder + video_name, 0, speed, (width,height))
    #
    # for image in images:
    #     video.write(cv2.imread(os.path.join(image_folder, image)))
    #
    # cv2.destroyAllWindows()
    # video.release()
