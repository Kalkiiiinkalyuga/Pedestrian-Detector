import tkinter as tk
from tkinter import filedialog
import cv2
import imutils
from PIL import Image, ImageTk

class PedestrianDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Detection App")

        self.label = tk.Label(root, text="Pedestrian Detection", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse Video File", command=self.browse_video_file)
        self.browse_button.pack(pady=10)

        self.detect_button = tk.Button(root, text="Detect Pedestrians", command=self.detect_pedestrians)
        self.detect_button.pack(pady=10)

        self.video_path_label = tk.Label(root, text="")
        self.video_path_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.pack(pady=10)

    def browse_video_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
        if file_path:
            self.video_path = file_path
            self.video_path_label.config(text=f"Selected Video File: {file_path}")

    def detect_pedestrians(self):
        if hasattr(self, 'video_path'):
            cap = cv2.VideoCapture(self.video_path)

            while cap.isOpened():
                ret, image = cap.read()
                if ret:
                    image = imutils.resize(image, width=min(400, image.shape[1]))

                    hog = cv2.HOGDescriptor()
                    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

                    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)

                    for (x, y, w, h) in regions:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                    # Display the output Image on Tkinter Canvas
                    self.display_image(image)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
        else:
            self.video_path_label.config(text="Please select a video file first.")

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Keep a reference to prevent garbage collection
        self.img_ref = img
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetectionApp(root)
    root.mainloop()
