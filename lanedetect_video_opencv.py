import matplotlib.pylab as plt
import cv2 as cv
import datetime
import imutils
import numpy as np
from tkinter import Tk, Label, Frame, StringVar, Entry, Button
import glob
import pickle

root = Tk()
root.geometry("600x400")
root.maxsize(650, 450)
root.minsize(550, 350)
root.title("GUI")
my = Label(text="Enter Video Path", bg='Blue',
                fg='White', font=('comicsansms', 19, 'bold'))
my.pack()
f = Frame(root, bg="grey", padx=50, pady=50)
user1 = StringVar()
user1.set("")
screen1 = Entry(f, textvar=user1, font='comicsansms 20 bold')
screen1.pack(pady=10)
f.pack(pady=10)


def main1():
    def undistort_img():
        # Stores all object points & img points from all images
        objpoints = []
        imgpoints = []

        # Prepare object points 0,0,0 ... 8,5,0
        obj_pts = np.zeros((6*9, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Get directory for all calibration images
        images = glob.glob('camera_cal/*.jpg')

        for indx, fname in enumerate(images):
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

            if ret:
                objpoints.append(obj_pts)
                imgpoints.append(corners)
        # Test undistortion on img
        img_size = (img.shape[1], img.shape[0])

        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            objpoints, imgpoints, img_size, None, None)

        dst = cv.undistort(img, mtx, dist, None, mtx)
        # Save camera calibration for later use
        dist_pickle = {}
        dist_pickle['mtx'] = mtx
        dist_pickle['dist'] = dist
        pickle.dump(dist_pickle, open('camera_cal/cal_pickle.p', 'wb'))

    undistort_img()

    def undistort(img, cal_dir='camera_cal/cal_pickle.p'):

        #cv2.imwrite('camera_cal/test_cal.jpg', dst)
        with open(cal_dir, mode='rb') as f:
            file = pickle.load(f)
        mtx = file['mtx']
        dist = file['dist']
        dst = cv.undistort(img, mtx, dist, None, mtx)

        return dst

    def draw_the_line(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2),
                        (0, 0, 255), thickness=2)
        img = cv.addWeighted(img, 0.8, blank_image, 1, 0.0)
        return img

    def region_of_interst(img, vertices):
        mask = np.zeros_like(img)
        # channel_count=img.shape[2]
        match_mask_color = 255
        cv.fillPoly(mask, vertices, match_mask_color)
        mask_image = cv.bitwise_and(img, mask)
        return mask_image

    def process(image):
        # print(image.shape)
        height = image.shape[0]
        width = image.shape[1]
        region_of_interst_vertices = [
            (0, height), (width / 2, height / 2), (width, height)]
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        canny = cv.Canny(gray, 250, 300)

        cropped = region_of_interst(canny, np.array(
            [region_of_interst_vertices], np.int32))

        line1 = cv.HoughLinesP(cropped, rho=10, theta=np.pi / 60, threshold=100, lines=np.array([]), minLineLength=5,
                               maxLineGap=5)

        image_width_line = draw_the_line(image, line1)
        return image_width_line

    def main(q):
        cap = cv.VideoCapture(q)
        fps_start_time = datetime.datetime.now()
        fps = 0
        total_frames = 0

        while(cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frame = process(frame)
                frame = undistort(frame)
                total_frames = total_frames + 1
                fps_end_time = datetime.datetime.now()
                time_diff = fps_end_time - fps_start_time

                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)

                fps_text = "FPS: {:.2f}".format(fps)
                cv.putText(frame, fps_text, (5, 30),
                           cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv.imshow('detection', frame)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                break

        cap.release()
        cv.destroyAllWindows()
    q = screen1.get()
    main(q)


b = Button(f, text="Enter", command=main1)
b.pack(pady=10)
root.mainloop()

