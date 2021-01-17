import matplotlib.pylab as plt
import cv2 as cv
import datetime
import imutils
import numpy as np
from tkinter import Tk, Label, Frame, StringVar, Entry, Button
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
    def draw_the_line(img, lines):
        img = np.copy(img)
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(blank_image, (x1, y1), (x2, y2),
                        (255, 255, 0), thickness=2)
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
            frame = process(frame)
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
        cap.release()
        cv.destroyAllWindows()
    q = screen1.get()
    main(q)


b = Button(f, text="Enter", command=main1)
b.pack(pady=10)
root.mainloop()
