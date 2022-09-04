import tkinter as tk  # This has all the code for GUIs.
# import tkinter.font as font      # This lets us use different fonts.
from tkinter import messagebox
import pymysql
# from PIL import ImageTk  # pip install pillow
# from datetime.tz import win
from PIL import ImageTk
from email_validation import check


# from email_validation import password_check
def center_window_on_screen():
    """
    This centres the window when it is not maximised.  It
    uses the screen and window height and width variables
    defined in the program below.
    :return: Nothing
    """

    root.geometry("1290x1080+100+50")


name555 = ""
email555 = ""
location555 = ""


def change_to_login():
    root.title("Login_Page")
    login_frame.pack(fill='both', expand=1)
    home_frame.forget()
    about_us_page.forget()
    helpline_page.forget()
    registration_page.forget()
    profile_page.forget()
    querie_page.forget()
    contact_us.forget()


def change_to_home():
    root.title("Home_Page")
    home_frame.pack(fill='both', expand=1)
    login_frame.forget()
    about_us_page.forget()
    helpline_page.forget()
    profile_page.forget()
    querie_page.forget()
    contact_us.forget()


def change_to_registration():
    root.title("Registration_Page")
    registration_page.pack(fill='both', expand=1)
    login_frame.forget()


def change_to_about_us():
    root.title("About_US_Page")
    about_us_page.pack(fill='both', expand=1)
    home_frame.forget()
    helpline_page.forget()
    profile_page.forget()
    querie_page.forget()
    contact_us.forget()


def change_to_profile():

    try:
        con = pymysql.connect(host="localhost", user="root", password="", database="mini_login")
        cur = con.cursor()
        # cur.execute("SELECT  FROM login WHERE username =" + us)
        cur.execute("select username, email, institute_name from login where username=%s and password = %s",
                    (username.get(), password.get()))
        # row = cur.fetchone()
        student = cur.fetchone()
        global name555
        name555 = student[0]
        global email555
        email555 = student[1]
        global location555
        location555 = student[2]

        con.close()
    except Exception as es:
        messagebox.showerror("Error", f"Error Due to : {str(es)}")
    root.title("Profile_Page")
    profile_page.pack(fill='both', expand=1)
    home_frame.forget()
    helpline_page.forget()
    about_us_page.forget()
    querie_page.forget()
    contact_us.forget()


def change_to_queries():
    root.title("Querie_Page")
    querie_page.pack(fill='both', expand=1)
    home_frame.forget()
    helpline_page.forget()
    about_us_page.forget()
    profile_page.forget()
    contact_us.forget()


def change_to_helpline_page():
    root.title("Helpline_Page")
    helpline_page.pack(fill='both', expand=1)
    home_frame.forget()
    about_us_page.forget()
    profile_page.forget()
    querie_page.forget()
    contact_us.forget()


def change_to_contact_us():
    root.title("Contact_Us_Page")
    contact_us.pack(fill='both', expand=1)
    home_frame.forget()
    about_us_page.forget()
    profile_page.forget()
    querie_page.forget()
    helpline_page.forget()


root = tk.Tk()
root.title("Face mask and Social Distance Tracker")
root.configure(bg='lightyellow')
# Set the icon used for your program
root.iconphoto(True,
               tk.PhotoImage(file="HAND.png"))

width, height = 500, 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_window_on_screen()
# number of frame as label
login_frame = tk.Frame(root)
home_frame = tk.Frame(root)
registration_page = tk.Frame(root)
about_us_page = tk.Frame(root)
helpline_page = tk.Frame(root)
profile_page = tk.Frame(root)
querie_page = tk.Frame(root)
contact_us = tk.Frame(root)
##### LOGIN PAGGE
# background image
bg = ImageTk.PhotoImage(file="social_dist.png")
bg_image = tk.Label(login_frame, image=bg)
bg_image.place(x=0, y=0, relwidth=1, relheight=1)
current_username = ""
current_password = ""


# database
def login():
    if username.get() == "" or password.get() == "":
        messagebox.showerror("Error", "Enter User Name And Password")
    else:
        try:
            con = pymysql.connect(host="localhost", user="root", password="", database="mini_login")
            cur = con.cursor()
            cur.execute("select * from login where username=%s and password = %s",
                        (username.get(), password.get()))
            row = cur.fetchone()
            if row is None:
                messagebox.showerror("Error", "Invalid User Name And Password")
                # home_secreen()
            else:
                messagebox.showinfo("Success", "Successfully Login")
                change_to_home()
                # close()
                # deshboard()
            con.close()
        except Exception as es:
            messagebox.showerror("Error", f"Error Due to : {str(es)}")


# ---------------------------------------------------------------End Login Function ---------------------------------
# login frame
Frame_login = tk.Frame(login_frame, bg="white")
Frame_login.place(x=330, y=150, width=500, height=400)

# title and subtitle
title = tk.Label(Frame_login, text="Login here", font=("Impact", 30, "bold"), fg="#B9BE77", bg="white")
title.place(x=90, y=30)
subtitle = tk.Label(Frame_login, text="Admin login area", font=("Goudy old style", 15, "bold"), fg="#A5678E",
                    bg="white")
subtitle.place(x=90, y=100)
# Username
lbl_user = tk.Label(Frame_login, text="Username", font=("Goudy old style", 15, "bold"), fg="grey",
                    bg="white")
lbl_user.place(x=90, y=140)
username = tk.Entry(Frame_login, font=("Goudy old style", 15), bg="#E7E6E6")
username.place(x=90, y=170, width=320, height=35)
# password
lbl_password = tk.Label(Frame_login, text="Password", font=("Goudy old style", 15, "bold"), fg="grey",
                        bg="white")
lbl_password.place(x=90, y=210)
# lbl_password.pack(pady=20)
password = tk.Entry(Frame_login, show="*", font=("Goudy old style", 15), bg="#E7E6E6")
password.place(x=90, y=240, width=320, height=35)
# password.pack(pady=20)
# Button
forget = tk.Button(Frame_login, text="Forgot Password?", bd=0, font=("Goudy old style", 12, "bold"), fg="#B9BE77",
                   bg="white")
forget.place(x=90, y=280)
# forget.pack(pady=20)
submit = tk.Button(Frame_login, text="Login?", bd=0, font=("Goudy old style", 15), bg="#B9BE77", fg="white",
                   command=login)
submit.place(x=90, y=320, width=150, height=40)
# submit.pack(pady=20)
Register = tk.Button(Frame_login, text="Register", bd=0, font=("Goudy old style", 15), bg="#B9BE77",
                     fg="white", command=change_to_registration)
Register.place(x=280, y=320, width=150, height=40)


# Register.pack(pady=20)
################ LOGIN PAGE DONE ###################


######################## HOME PAGE START $$$$$$$$$$$$$$$$$$$$
def face_detector():
    home_frame.forget()
    # import the necessary packages
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
    from imutils.video import VideoStream
    import numpy as np
    import imutils
    # import time
    import cv2
    # import os

    def detect_and_predict_mask(frame11, faceNet11, maskNet11):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame11.shape[:2]
        blob = cv2.dnn.blobFromImage(frame11, 1.0, (224, 224),
                                     (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the face detections
        faceNet11.setInput(blob)
        detections = faceNet11.forward()
        print(detections.shape)

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs11 = []
        preds11 = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box11 = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX1, startY1, endX1, endY1) = box11.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX1, startY1) = (max(0, startX1), max(0, startY1))
                (endX1, endY1) = (min(w - 1, endX1), min(h - 1, endY1))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame11[startY1:endY1, startX1:endX1]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs11.append((startX1, startY1, endX1, endY1))

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds11 = maskNet11.predict(faces, batch_size=32)

        # return a 2-tuple of the face locations and their corresponding
        # locations
        return locs11, preds11

    # load our serialized face detector model from disk
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    # initialize the video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=1290, height=1080, )

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask11, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask11 > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            violate = set()
            if label == "No Mask":
                violate.add(1)
                text = "Number of Unmask person: {}".format(len(violate))
                cv2.putText(frame, text, (10, frame.shape[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask11, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            change_to_home()
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    ### face mask function end#####################


def social_distance():
    from mylib import config, thread
    from mylib.mailer import Mailer
    from mylib.detection import detect_people
    from imutils.video import VideoStream, FPS
    from scipy.spatial import distance as dist
    import numpy as np
    import argparse, imutils, cv2, os, time, schedule

    # ----------------------------Parse req. arguments------------------------------#
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="",
                    help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="",
                    help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1,
                    help="whether or not output frame should be displayed")
    args = vars(ap.parse_args())
    # ------------------------------------------------------------------------------#

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # check if we are going to use GPU
    if config.USE_GPU:
        # set CUDA as the preferable backend and target
        print("")
        print("[INFO] Looking for GPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # if a video path was not supplied, grab a reference to the camera
    if not args.get("input", False):
        print("[INFO] Starting the live stream..")
        vs = cv2.VideoCapture(0)
        if config.Thread:
            cap = thread.ThreadingClass(config.url)
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] Starting the video..")
        vs = cv2.VideoCapture(args["input"])
        if config.Thread:
            cap = thread.ThreadingClass(args["input"])

    writer = None
    # start the FPS counter
    fps = FPS().start()

    # loop over the frames from the video stream
    while True:
        # read the next frame from the file
        if config.Thread:
            frame = cap.read()

        else:
            (grabbed, frame) = vs.read()
            # if the frame was not grabbed, then we have reached the end of the stream
            if not grabbed:
                break

        # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, net, ln,
                                personIdx=LABELS.index("person"))

        # initialize the set of indexes that violate the max/min social distance limits
        serious = set()
        abnormal = set()

        # ensure there are *at least* two people detections (required in
        # order to compute our pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the
            # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update our violation set with the indexes of the centroid pairs
                        serious.add(i)
                        serious.add(j)
                    # update our abnormal set if the centroid distance is below max distance limit
                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                        abnormal.add(i)
                        abnormal.add(j)

        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then
            # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation/abnormal sets, then update the color
            if i in serious:
                color = (0, 0, 255)
            elif i in abnormal:
                color = (0, 255, 255)  # orange = (0, 165, 255)

            # draw (1) a bounding box around the person and (2) the
            # centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 2)

        # draw some of the parameters
        Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
        cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
        Threshold = "Threshold limit: {}".format(config.Threshold)
        cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)

        # draw the total number of social distancing violations on the output frame
        text = "Total serious violations: {}".format(len(serious))
        cv2.putText(frame, text, (10, frame.shape[0] - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

        text1 = "Total abnormal violations: {}".format(len(abnormal))
        cv2.putText(frame, text1, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

        # ------------------------------Alert function----------------------------------#
        if len(serious) >= config.Threshold:
            cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
                        cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
            if config.ALERT:
                print("")
                print('[INFO] Sending mail...')
                Mailer().send(config.MAIL)
                print('[INFO] Mail sent')
        # config.ALERT = False
        # ------------------------------------------------------------------------------#
        # check to see if the output frame should be displayed to our screen
        if args["display"] > 0:
            # show the output frame
            cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
        # update the FPS counter
        fps.update()

        # if an output video file path has been supplied and the video
        # writer has not been initialized, do so now
        if args["output"] != "" and writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output video file
        if writer is not None:
            writer.write(frame)

    # stop the timer and display FPS information
    fps.stop()
    print("===========================")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    # close any open windows
    cv2.destroyAllWindows()


# background color
Frame_home = tk.Frame(home_frame, bg="#EDECF2")
Frame_home.place(width=1920, height=1080)
# Frame_home.pack(pady=20)

t1 = tk.Label(Frame_home, text="Welcome to face mask and social distance detector", font=("Arial", 15), fg="#8D91AA",
              bg="#EDECF2").place(x=600, y=150)
t2 = tk.Label(Frame_home,
              text="We encourage you all to wear a mask while going out to help stop the spread of coronavirus",
              font=("Arial", 15), fg="#8D91AA", bg="#EDECF2").place(x=480, y=180)
t3 = tk.Label(Frame_home, text="Please maintain social distance wherever you go for safety purpose", font=("Arial", 15),
              fg="#8D91AA", bg="#EDECF2").place(x=480, y=210)

# Home first photo
log = ImageTk.PhotoImage(file="HAND.png")
log_image = tk.Label(home_frame, image=log).place(x=320, y=300, width=260, height=371)

# Home second photo
socdist = ImageTk.PhotoImage(file="SD.jpeg")
socdist_image = tk.Label(home_frame, image=socdist).place(x=600, y=300, width=260, height=371)

# Home third photo
mask = ImageTk.PhotoImage(file="mask.jpeg")
mask_image = tk.Label(home_frame, image=mask).place(x=880, y=300, width=260, height=371)

# logo
logo2 = ImageTk.PhotoImage(file="Dtector.png")
logo_image = tk.Label(home_frame, image=logo2)
logo_image.place(x=0, y=0)
# logo_image.pack()
Frame_home = tk.Frame(home_frame, bg="#C3CACE")
Frame_home.place(x=0, y=134, width=300, height=941)

# profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#754B61",
#                   fg="white", command=change_to_profile)
# profile.place(x=50, y=30, width=170, height=40)
# profile.pack()
About = tk.Button(Frame_home, text="About us", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                  command=change_to_about_us)
About.place(
    x=50, y=80, width=170, height=40)
# About.pack()
Queries = tk.Button(Frame_home, text="Queries", bd=0, font=("Goudy old style", 15), bg="#754B61",
                    fg="white", command=change_to_queries)
Queries.place(x=50, y=130, width=170, height=40)
# Queries.pack()
Help = tk.Button(Frame_home, text="Helpline", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                 command=change_to_helpline_page)
Help.place(
    x=50, y=180, width=170, height=40)
# horizontal frame
Frame_home = tk.Frame(home_frame, bg="#4C3949")
Frame_home.place(x=159, y=0, width=1761, height=60)

title = tk.Label(Frame_home, text="Face Mask & Social Distance Detector", font=("Arial", 15, "bold"), fg="white",
                 bg="#4C3949")
title.place(x=300, y=10)
# hor frame 2
Frame_home = tk.Frame(home_frame, bg="#B2ABB3")
Frame_home.place(x=159, y=60, width=1761, height=70)
# Frame_home.pack()
hm = tk.Button(Frame_home, text="Home", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="red")
hm.place(x=50, y=10, width=140, height=40)
# hm.pack()
Face = tk.Button(Frame_home, text="Face Mask Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                 fg="Black", command=face_detector)
Face.place(x=210, y=10, width=180, height=40)
# Face.pack()
Social = tk.Button(Frame_home, text="Social Distance Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=social_distance)
Social.place(x=410, y=10, width=220, height=40)
# Social.pack(pady=20)
Contact = tk.Button(Frame_home, text="Contact Us", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                    fg="Black", command=change_to_contact_us)
Contact.place(x=650, y=10, width=180, height=40)
# logout
logout = tk.Button(Frame_home, text="Logout", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=change_to_login)
logout.place(x=850, y=10, width=180, height=40)
# Contact.pack(pady=20)
# PROFILE
profile = tk.Button(Frame_home, text="HE I AM HERE", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                    fg="white", )
profile.place(x=50, y=334, width=150, height=40)


####################### HOME PAGE END ####################


####################### REGISTRATION PAGE START ############################


######## Registration data base #######################################
# signup database connect
def action():
    if username_1.get() == "" or password_1.get() == "" or confirm_password_1.get() == "" or c1_v1.get() == 'No' or email_id_1.get() == "" or institute_name_1.get() == "":
        messagebox.showerror("Error", "All Fields Are Required")
    elif password_1.get() != confirm_password_1.get():
        messagebox.showerror("Error", "Password & Confirm Password Should Be Same")
    elif not check(email_id_1.get()):
        messagebox.showerror("Error", "Entered Email is Not Valid ")
    elif not validation():
        pass
    else:
        try:
            con = pymysql.connect(host="localhost", user="root", password="", database="mini_login")
            cur = con.cursor()
            cur.execute("select * from login where institute_name=%s", institute_name_1.get())
            row = cur.fetchone()
            if row is not None:
                messagebox.showerror("Error", "Institute Name Already Exists")
            
            else:
                cur.execute(
                    "insert into login(username,password,email,institute_name) values(%s,%s,%s,%s)",
                    (
                        username_1.get(),
                        password_1.get(),
                        email_id_1.get(),
                        institute_name_1.get()
                    ))
                con.commit()
                con.close()
                messagebox.showinfo("Success", "Registration Successful")
                print('Check box value :', c1_v1.get())
        except Exception as es:
            messagebox.showerror("Error", f"Error Dui to : {str(es)}")


special_ch = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|',
              '\\', '/', ':', ';', '"', "'", '<', '>', ',', '.', '?']


def validation():
    password1 = password_1.get()
    msg = ""
    check12 = False
    if len(password1) == 0:
        msg = 'Password can\'t be empty'
    else:
        try:
            if not any(ch in special_ch for ch in password1):
                msg = 'At least 1 special character required!'
            elif not any(ch.isupper() for ch in password1):
                msg = 'At least 1 uppercase character required!'
            elif not any(ch.islower() for ch in password1):
                msg = 'At least 1 lowercase character required!'
            elif not any(ch.isdigit() for ch in password1):
                msg = 'At least 1 number required!'
            elif len(password1) < 4:
                msg = 'Password must be minimum of 4 characters!'
            else:
                return True
        except Exception as ep:
            messagebox.showerror('error', ep)
    messagebox.showinfo('message', msg)
    return check12


# background image
bg_1 = ImageTk.PhotoImage(file="social_dist.png")
bg_image_1 = tk.Label(registration_page, image=bg_1)
bg_image_1.place(x=0, y=0, relwidth=1, relheight=1)

# Registration frame
Frame_register = tk.Frame(registration_page, bg="white")
Frame_register.place(x=330, y=30, width=500, height=700)

# title and subtitle
title_1 = tk.Label(Frame_register, text="Register here", font=("Impact", 30, "bold"), fg="#B9BE77", bg="white")
title_1.place(x=90, y=30)
subtitle_1 = tk.Label(Frame_register, text="For new users ..!!", font=("Goudy old style", 15, "bold"), fg="#A5678E",
                      bg="white")
subtitle_1.place(x=90, y=80)
# Username
lbl_user_1 = tk.Label(Frame_register, text="Username", font=("Goudy old style", 15, "bold"), fg="grey", bg="white")
lbl_user_1.place(x=90, y=110)
username_1 = tk.Entry(Frame_register, font=("Goudy old style", 15), bg="#E7E6E6")
username_1.place(x=90, y=140, width=320, height=30)

# password
lbl_password_1 = tk.Label(Frame_register, text="Password", font=("Goudy old style", 15, "bold"), fg="grey", bg="white")
lbl_password_1.place(x=90, y=170)
password_1 = tk.Entry(Frame_register, show="*", font=("Goudy old style", 15), bg="#E7E6E6")
password_1.place(x=90, y=200, width=320, height=30)

# Confirm password
lbl_confirm_password_1 = tk.Label(Frame_register, text="Confirm Password", font=("Goudy old style", 15, "bold"),
                                  fg="grey",
                                  bg="white")
lbl_confirm_password_1.place(x=90, y=240)
confirm_password_1 = tk.Entry(Frame_register, show="*", font=("Goudy old style", 15), bg="#E7E6E6")
confirm_password_1.place(x=90, y=280, width=320, height=30)

# email id
lbl_email_id_1 = tk.Label(Frame_register, text="Email id", font=("Goudy old style", 15, "bold"), fg="grey", bg="white")
lbl_email_id_1.place(x=90, y=320)
email_id_1 = tk.Entry(Frame_register, font=("Goudy old style", 15), bg="#E7E6E6")
email_id_1.place(x=90, y=360, width=320, height=30)

# Institute name
lbl_institute_name_1 = tk.Label(Frame_register, text="Institute Name", font=("Goudy old style", 15, "bold"), fg="grey",
                                bg="white")
lbl_institute_name_1.place(x=90, y=400)
institute_name_1 = tk.Entry(Frame_register, font=("Goudy old style", 15), bg="#E7E6E6")
institute_name_1.place(x=90, y=440, width=320, height=30)

import tkinter as tk


# terms and conditions

def my_upd():
    print('Check box value :', c1_v1.get())


c1_v1 = tk.StringVar()
c1_v1.set('No')
c1_1 = tk.Checkbutton(Frame_register, text="Accept terms and conditions", variable=c1_v1,
                      onvalue='Yes', offvalue='No')
c1_1.place(x=80, y=480)
# Button
submit_1 = tk.Button(Frame_register, text="Register", bd=0, font=("Goudy old style", 15), bg="#B9BE77", fg="white",
                     command=action)
submit_1.place(x=60, y=520, width=100, height=40)
already_1 = tk.Button(Frame_register, text="Already registered? Login>>", bd=0, font=("Goudy old style", 15),
                      bg="#B9BE77",
                      fg="white", command=change_to_login)
already_1.place(x=230, y=520, width=230, height=40)

####################### REGISTRATION PAGE END ############################

####################### ABOUT_US_PAGE START #############################

root.geometry("1920x1080+100+50")
# background color
Frame_home = tk.Frame(about_us_page, bg="#EDECF2")
Frame_home.place(width=1920, height=1080)
# line = Label(Frame_home, text="", font=("Impact", 15, "bold"), fg="purple", bg="#EDECF2").place(x= 470, y=200)

# central frame 2
Frame_home = tk.Frame(about_us_page, bg="#EDECF2")
Frame_home.place(x=310, y=180, width=1050, height=800)

title = tk.Label(Frame_home,
                 text="Don't let Social Distance lead to emotional distance. Reach out, open up and let people in. ",
                 font=("Impact", 15), fg="#754B61", bg="#EDECF2").place(x=90, y=30)
t1 = tk.Label(Frame_home, text="Reach out, open up and let people in. ", font=("Impact", 15), fg="#754B61",
              bg="#EDECF2").place(x=90, y=70)
t2 = tk.Label(Frame_home,
              text=" We may not be able to be with each other right now, but we can still be there for each other. ",
              font=("Impact", 15), fg="#754B61", bg="#EDECF2").place(x=90, y=110)

# Home first photo
log3 = ImageTk.PhotoImage(file="mask.jpg")
log_image = tk.Label(about_us_page, image=log3)
log_image.place(x=320, y=350, width=188, height=176)
clean = ImageTk.PhotoImage(file="cleanse.jpg")
clean_image = tk.Label(about_us_page, image=clean)
clean_image.place(x=550, y=350, width=188, height=176)
wash = ImageTk.PhotoImage(file="wash.jpg")
wash_image = tk.Label(about_us_page, image=wash)
wash_image.place(x=790, y=350, width=188, height=176)
dis = ImageTk.PhotoImage(file="dis.jpg")
dis_image = tk.Label(about_us_page, image=dis)
dis_image.place(x=1050, y=350, width=188, height=176)
# centre frame
Frame_home = tk.Frame(about_us_page, bg="#DCE2EC")
Frame_home.place(x=310, y=134, width=1050, height=50)

# profile header
title = tk.Label(Frame_home, text="About us...", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180, y=10)

# color code
t1 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#4C3949").place(x=70, y=10)
t2 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#754B61").place(x=90, y=10)
t3 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#B2ABB3").place(x=110, y=10)
t4 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#C3CACE").place(x=130, y=10)

# logo
logo3 = ImageTk.PhotoImage(file="Dtector.png")
logo_image = tk.Label(about_us_page, image=logo3).place(x=0, y=0)

# Standing frame
Frame_home = tk.Frame(about_us_page, bg="#C3CACE")
Frame_home.place(x=0, y=134, width=300, height=941)
# profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
#                   command=change_to_profile).place(
#  x=50,
# y=30,
# width=170,
# height=40)
About = tk.Button(Frame_home, text="About us", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="red",
                  command=change_to_about_us).place(x=50,
                                                    y=80,
                                                    width=170,
                                                    height=40)
Queries = tk.Button(Frame_home, text="Queries", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                    command=change_to_queries).place(
    x=50,
    y=130,
    width=170,
    height=40)
Help = tk.Button(Frame_home, text="Helpline", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                 command=change_to_helpline_page).place(x=50,
                                                        y=180,
                                                        width=170,
                                                        height=40)

# horizontal frame
Frame_home = tk.Frame(about_us_page, bg="#4C3949")
Frame_home.place(x=159, y=0, width=1761, height=60)
title = tk.Label(Frame_home, text="Face Mask & Social Distance Detector", font=("Arial", 15, "bold"), fg="white",
                 bg="#4C3949").place(x=300, y=10)

# hor frame 2
Frame_home = tk.Frame(about_us_page, bg="#B2ABB3")
Frame_home.place(x=159, y=60, width=1761, height=70)
hm = tk.Button(Frame_home, text="Home", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
               command=change_to_home).place(x=50, y=10,
                                             width=140,
                                             height=40)
Face = tk.Button(Frame_home, text="Face Mask Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                 fg="Black", command=face_detector)
Face.place(x=210, y=10, width=180, height=40)
# Face.pack()
Social = tk.Button(Frame_home, text="Social Distance Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=social_distance)
Social.place(x=410, y=10, width=220, height=40)
Contact = tk.Button(Frame_home, text="Contact Us", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
                    command=change_to_contact_us).place(
    x=650, y=10, width=180, height=40)
logout = tk.Button(Frame_home, text="Logout", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=change_to_login)
logout.place(x=850, y=10, width=180, height=40)

# PROFILE
profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="white").place(
    x=50,
    y=334,
    width=150,
    height=40)

####################### ABHOUT_US_PAGE END #############################


####################### HELPLINE PAGE START###########################3#

Frame_home = tk.Frame(helpline_page, bg="#EDECF2")
Frame_home.place(width=1920, height=1080)
# line = Label(Frame_home, text="", font=("Impact", 15, "bold"), fg="purple", bg="#EDECF2").place(x= 470, y=200)


# centre frame
Frame_home = tk.Frame(helpline_page, bg="#DCE2EC")
Frame_home.place(x=310, y=134, width=1050, height=50)

# profile header
title = tk.Label(Frame_home, text="HELPLINES", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180, y=10)

# color code
t1 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#4C3949").place(x=70, y=10)
t2 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#754B61").place(x=90, y=10)
t3 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#B2ABB3").place(x=110, y=10)
t4 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#C3CACE").place(x=130, y=10)

# central frame 2
Frame_home = tk.Frame(helpline_page, bg="white")
Frame_home.place(x=310, y=200, width=1050, height=500)
Frame_home = tk.Frame(helpline_page, bg="#DCE2EC")
Frame_home.place(x=400, y=250, width=900, height=400)
help12 = ImageTk.PhotoImage(file="help1.png")
help1_image = tk.Label(helpline_page, image=help12).place(x=900, y=310)

title = tk.Label(Frame_home, text="COVID HELPLINES ", font=("Impact", 15), fg="#754B61", bg="#EDECF2").place(x=160,
                                                                                                             y=30)
t5 = tk.Label(Frame_home, text="CORONA (COVID 19) HELPLINE : 1075 OR 011-23978046 ", font=("Impact", 15), fg="#754B61",
              bg="#DCE2EC").place(x=30, y=120)
t6 = tk.Label(Frame_home, text=" MAHARASHTRA COVID HELPLINE:  ", font=("Impact", 15), fg="#754B61", bg="#DCE2EC").place(
    x=30, y=160)
t7 = tk.Label(Frame_home, text=" 022 - 22027990020 - 26127394 ", font=("Impact", 15), fg="#754B61", bg="#DCE2EC").place(
    x=30, y=190)
t8 = tk.Label(Frame_home, text=" COVID ENQUIRY:  MAHAINFOCORONA.IN ", font=("Impact", 15), fg="#754B61",
              bg="#DCE2EC").place(x=30, y=240)
# logo
logo233 = ImageTk.PhotoImage(file="Dtector.png")
logo_image = tk.Label(helpline_page, image=logo233).place(x=0, y=0)

# Standing frame
Frame_home = tk.Frame(helpline_page, bg="#C3CACE")
Frame_home.place(x=0, y=134, width=300, height=941)
# profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
#                   command=change_to_profile).place(
#  x=50, y=30, width=170, height=40)
About49 = tk.Button(Frame_home, text="About us", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                    command=change_to_about_us).place(x=50, y=80, width=170, height=40)
Queries48 = tk.Button(Frame_home, text="Queries", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                      command=change_to_queries).place(
    x=50, y=130, width=170, height=40)
Help47 = tk.Button(Frame_home, text="Helpline", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="red").place(x=50,
                                                                                                                  y=180,
                                                                                                                  width=170,
                                                                                                                  height=40)

# horizontal frame
Frame_home = tk.Frame(helpline_page, bg="#4C3949")
Frame_home.place(x=159, y=0, width=1761, height=60)
title46 = tk.Label(Frame_home, text="Face Mask & Social Distance Detector", font=("Arial", 15, "bold"), fg="white",
                   bg="#4C3949").place(x=300, y=10)

# hor frame 2
Frame_home = tk.Frame(helpline_page, bg="#B2ABB3")
Frame_home.place(x=159, y=60, width=1761, height=70)
hm45 = tk.Button(Frame_home, text="Home", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
                 command=change_to_home).place(x=50, y=10, width=140, height=40)
Face = tk.Button(Frame_home, text="Face Mask Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                 fg="Black", command=face_detector)
Face.place(x=210, y=10, width=180, height=40)
# Face.pack()
Social = tk.Button(Frame_home, text="Social Distance Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=social_distance)
Social.place(x=410, y=10, width=220, height=40)

Contact42 = tk.Button(Frame_home, text="Contact Us", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                      fg="Black", command=change_to_contact_us).place(x=650, y=10, width=180, height=40)
logout = tk.Button(Frame_home, text="Logout", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=change_to_login)
logout.place(x=850, y=10, width=180, height=40)
# PROFILE
# profile41 = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="white").place(
#   x=50, y=334, width=150, height=40)

####################### HELPLINE PAGE END###########################3#


################################### QUERIES PAGE START #####################################################

# background color
Frame_home = tk.Frame(querie_page, bg="#EDECF2")
Frame_home.place(width=1920, height=1080)
# line = Label(Frame_home, text="", font=("Impact", 15, "bold"), fg="purple", bg="#EDECF2").place(x= 470, y=200)


# logo
logo = ImageTk.PhotoImage(file="Dtector.png")
logo_image = tk.Label(querie_page, image=logo).place(x=0, y=0)

# Standing frame
Frame_home = tk.Frame(querie_page, bg="#C3CACE")
Frame_home.place(x=0, y=134, width=300, height=941)
# profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
#                   command=change_to_profile).place(x=50, y=30, width=170, height=40)
About = tk.Button(Frame_home, text="About us", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                  command=change_to_about_us).place(x=50, y=80, width=170, height=40)
Queries = tk.Button(Frame_home, text="Queries", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="red").place(x=50,
                                                                                                                  y=130,
                                                                                                                  width=170,
                                                                                                                  height=40)
Help = tk.Button(Frame_home, text="Helpline", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                 command=change_to_helpline_page).place(x=50, y=180, width=170, height=40)

# horizontal frame
Frame_home = tk.Frame(querie_page, bg="#4C3949")
Frame_home.place(x=159, y=0, width=1761, height=60)
title = tk.Label(Frame_home, text="Face Mask & Social Distance Detector", font=("Arial", 15, "bold"), fg="white",
                 bg="#4C3949").place(x=300, y=10)

# hor frame 2
Frame_home = tk.Frame(querie_page, bg="#B2ABB3")
Frame_home.place(x=159, y=60, width=1761, height=70)
hm = tk.Button(Frame_home, text="Home", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
               command=change_to_home).place(x=50, y=10, width=140, height=40)
Face = tk.Button(Frame_home, text="Face Mask Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                 fg="Black", command=face_detector)
Face.place(x=210, y=10, width=180, height=40)
# Face.pack()
Social = tk.Button(Frame_home, text="Social Distance Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=social_distance)
Social.place(x=410, y=10, width=220, height=40)
Contact = tk.Button(Frame_home, text="Contact Us", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
                    command=change_to_contact_us).place(x=650, y=10, width=180, height=40)
logout = tk.Button(Frame_home, text="Logout", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=change_to_login)
logout.place(x=850, y=10, width=180, height=40)
# centre frame
Frame_home = tk.Frame(querie_page, bg="#DCE2EC")
Frame_home.place(x=310, y=134, width=1050, height=50)
# profile header
t8 = tk.Label(Frame_home, text="QUERIES", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180, y=10)

# color code
t1 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#4C3949").place(x=70, y=10)
t2 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#754B61").place(x=90, y=10)
t3 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#B2ABB3").place(x=110, y=10)
t4 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#C3CACE").place(x=130, y=10)

# central frame 3
Frame_home = tk.Frame(querie_page, bg="#DCE2EC")
Frame_home.place(x=310, y=190, width=500, height=600)
t8 = tk.Label(Frame_home, text="Face Mask Detector", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180, y=10)
f1 = tk.Label(Frame_home, text="Step 1: Click on FACE MASK DETECTOR", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=50)
f2 = tk.Label(Frame_home, text="Step 2: Sit at appropriate distance with ur face clear.", font=("Arial", 15),
              fg="#8D91AA", bg="#DCE2EC").place(x=30, y=80)
f3 = tk.Label(Frame_home, text="Step 3: Green shows mask detected", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=120)
f4 = tk.Label(Frame_home, text="Step 4: Red box shows no mask.", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(
    x=30, y=160)
# central frame 4
Frame_home = tk.Frame(querie_page, bg="#DCE2EC")
Frame_home.place(x=850, y=190, width=500, height=600)
t9 = tk.Label(Frame_home, text="Social Distance Detector", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180,
                                                                                                                 y=10)
s1 = tk.Label(Frame_home, text="Step 1: Click on SOCIAL DISTANCE DETECTOR", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=50)
s2 = tk.Label(Frame_home, text="Step 2: Sit/Walk at appropriate distance.", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=80)
s3 = tk.Label(Frame_home, text="Step 3: Green shows distance detected", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=120)
s4 = tk.Label(Frame_home, text="Step 4: Red box shows social distance violated.", font=("Arial", 15), fg="#8D91AA",
              bg="#DCE2EC").place(x=30, y=160)


###################################################### QUERIES PAGE END #################################################################

###################################################### CONTACT US PAGE START #############################################################
def CONTACT():
    name2 = name.get()
    email1 = email3.get()
    phone4 = phone3.get()
    mess4 = mess3.get()
    try:
        con = pymysql.connect(host="localhost", user="root", password="", database="mini_login")
        cur = con.cursor()
        cur.execute("select * from contact where name=%s", name2)
        cur.execute(
            "insert into contact(name,email,phone,message) values(%s,%s,%s,%s)",
            (
                name2,
                email1,
                phone4,
                mess4
            ))
        con.commit()
        con.close()
        messagebox.showinfo("Success",
                            "Your message is received successfully, we will revert back to you through mail!")
        print('Check box value :', c1_v1.get())
    except Exception as es:
        messagebox.showerror("Error", f"Error Dui to : {str(es)}")


# background color
Frame_home = tk.Frame(contact_us, bg="#EDECF2")
Frame_home.place(width=1920, height=1080)
# line = Label(Frame_home, text="", font=("Impact", 15, "bold"), fg="purple", bg="#EDECF2").place(x= 470, y=200)

# centre frame
Frame_home = tk.Frame(contact_us, bg="#DCE2EC")
Frame_home.place(x=310, y=134, width=1050, height=50)

# profile header
title = tk.Label(Frame_home, text="Contact us", font=("Arial", 15), fg="#8D91AA", bg="#DCE2EC").place(x=180, y=10)

# color code
t1 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#4C3949").place(x=70, y=10)
t2 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#754B61").place(x=90, y=10)
t3 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#B2ABB3").place(x=110, y=10)
t4 = tk.Label(Frame_home, text="  ", fg="#8D91AA", bg="#C3CACE").place(x=130, y=10)

# logo
logo69 = ImageTk.PhotoImage(file="Dtector.png")
logo_image = tk.Label(contact_us, image=logo69).place(x=0, y=0)

# Standing frame
Frame_home = tk.Frame(contact_us, bg="#C3CACE")
Frame_home.place(x=0, y=134, width=300, height=941)
# profile = tk.Button(Frame_home, text="Profile", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
#                  command=change_to_profile).place(x=50, y=30, width=170, height=40)
About = tk.Button(Frame_home, text="About us", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                  command=change_to_about_us).place(x=50, y=80, width=170, height=40)
Queries = tk.Button(Frame_home, text="Queries", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                    command=change_to_queries).place(x=50, y=130, width=170, height=40)
Help = tk.Button(Frame_home, text="Helpline", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                 command=change_to_helpline_page).place(x=50, y=180, width=170, height=40)

# horizontal frame
Frame_home = tk.Frame(contact_us, bg="#4C3949")
Frame_home.place(x=159, y=0, width=1761, height=60)
title = tk.Label(Frame_home, text="Face Mask & Social Distance Detector", font=("Arial", 15, "bold"), fg="white",
                 bg="#4C3949").place(x=300, y=10)

# hor frame 2
Frame_home = tk.Frame(contact_us, bg="#B2ABB3")
Frame_home.place(x=159, y=60, width=1761, height=70)
hm = tk.Button(Frame_home, text="Home", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="Black",
               command=change_to_home).place(x=50, y=10, width=140, height=40)
Face = tk.Button(Frame_home, text="Face Mask Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                 fg="Black", command=face_detector)
Face.place(x=210, y=10, width=180, height=40)
# Face.pack()
Social = tk.Button(Frame_home, text="Social Distance Detector", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=social_distance)
Social.place(x=410, y=10, width=220, height=40)
Contact = tk.Button(Frame_home, text="Contact Us", bd=0, font=("Goudy old style", 15), bg="#C3CACE", fg="red").place(
    x=650, y=10, width=180, height=40)
logout = tk.Button(Frame_home, text="Logout", bd=0, font=("Goudy old style", 15), bg="#C3CACE",
                   fg="Black", command=change_to_login)
logout.place(x=850, y=10, width=180, height=40)

# central frame
Frame_home = tk.Frame(contact_us, bg="white")
Frame_home.place(x=310, y=200, width=1050, height=500)
# name
lbl_name = tk.Label(Frame_home, text="Name:", font=("Goudy old style", 15, "bold"), fg="grey", bg="white").place(x=90,
                                                                                                                 y=70)
name = tk.Entry(Frame_home, font=("Goudy old style", 15), bg="#E7E6E6")
name.place(x=200, y=70, width=320, height=35)

# email
lbl_email = tk.Label(Frame_home, text="Email:", font=("Goudy old style", 15, "bold"), fg="grey", bg="white").place(x=90,
                                                                                                                   y=130)
email3 = tk.Entry(Frame_home, font=("Goudy old style", 15), bg="#E7E6E6")
email3.place(x=200, y=130, width=320, height=35)

# phone
lbl_phone = tk.Label(Frame_home, text="Phone No.:", font=("Goudy old style", 15, "bold"), fg="grey", bg="white").place(
    x=90, y=190)
phone3 = tk.Entry(Frame_home, font=("Goudy old style", 15), bg="#E7E6E6")
phone3.place(x=200, y=190, width=320, height=35)

# message
lbl_mess = tk.Label(Frame_home, text="Message:", font=("Goudy old style", 15, "bold"), fg="grey", bg="white").place(
    x=90, y=260)
mess3 = tk.Entry(Frame_home, font=("Goudy old style", 15), bg="#E7E6E6")
mess3.place(x=200, y=260, width=320, height=75)

# submit
submit = tk.Button(Frame_home, text="Submit", bd=0, font=("Goudy old style", 15), bg="#754B61", fg="white",
                   command=CONTACT).place(x=110,
                                          y=380,
                                          width=150,
                                          height=40)
contact = ImageTk.PhotoImage(file="help1.png")
contact_image = tk.Label(contact_us, image=contact).place(x=900, y=270)

###################################################### CONTACT US PAGE END #############################################################


####################### END ###################
login_frame.pack(fill='both', expand=1)
root.mainloop()
