import cv2
from flask import *
import mediapipe as mp
import numpy as np
from keras.models import load_model
from datetime import datetime
import time
import os
import math
from passlib.hash import sha256_crypt
from mysql.connector import connect
import pandas as pd

app = Flask(__name__)
app.secret_key = 'b666a068bc03bf92994fe986cc8abbf1'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'virtualgym'

db = connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

user_label = ['username' ,'email' ,'password' ,'full_name' ,'age' ,'height' ,'weight' ,'gender']
bicep_rep_data_label = ['exer_name', 'l_reps', 'r_reps', 'rh_reps', 'lh_reps', 'rl_elbow', 'll_elbow', 'timestamp', 'username', 'start_timestamp']
squats_rep_label = ['exer_name', 'rep', 'timestamp', 'username', 'start_timestamp']


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index4.html')

##custom tags
@app.template_filter('time_taken')
def time_taken(start, end):
    elapsed = end-start
    return int(elapsed.total_seconds())


@app.template_filter('date_conv')
def date_conv(times):
    return times.strftime("%Y-%m-%d")






##routes


@app.route('/')
def index2():
    return render_template('index2.html')

@app.route('/index1/')
def index1():
    return render_template('index1.html')

@app.route('/BicepCurl/')
def BicepCurl():
    return render_template('BicepCurl.html')

@app.route('/YogaPose/')
def YogaPose():
    return render_template('YogaPose.html')

@app.route('/Squats/')
def Squats():
    return render_template('Squats.html')

@app.route('/Services/')
def Services():
    return render_template('Services.html')

@app.route('/profile/')
def profile():
    username = session['username']
    cursor = db.cursor()
    cursor.execute('SELECT * FROM user WHERE username=%s', (username, ))
    user = cursor.fetchall()
    cursor.execute('SELECT * FROM biceprepdata WHERE username=%s', (username, ))
    bicep_data = cursor.fetchall()
    cursor.execute("SELECT * FROM squatsrepdata WHERE username=%s ORDER BY 'timestamp' DESC LIMIT 10", (username, ))
    squats_data = cursor.fetchall()
    user = pd.DataFrame(user, columns=user_label)
    bicep_data = pd.DataFrame(bicep_data, columns=bicep_rep_data_label)
    squats_data = pd.DataFrame(squats_data, columns=squats_rep_label)
    
    cursor.close()
    
    squats_end_date = squats_data['timestamp'].to_list()
    squats_start_date = squats_data['start_timestamp'].to_list()
    squats_graph_ydata = []
    for i in range(0, len(squats_end_date)):
        squats_graph_ydata.append(time_taken(squats_start_date[i], squats_end_date[i]))
    squats_graph_xdata = [i.strftime("%Y-%m-%d") for i in squats_end_date]
        
    l_reps = bicep_data['l_reps'].to_list()
    r_reps = bicep_data['r_reps'].to_list()
    rh_reps = bicep_data['rh_reps'].to_list()
    lh_reps = bicep_data['lh_reps'].to_list()
    rl_elbow = bicep_data['rl_elbow'].to_list()
    ll_elbow = bicep_data['ll_elbow'].to_list()
    b_timestamp = [i.strftime("%Y-%m-%d") for i in bicep_data['timestamp'].to_list()]
    
    return render_template('profile.html', user=user, bicep_data=bicep_data, squats_data=squats_data, 
                           squats_graph_ydata=squats_graph_ydata, squats_graph_xdata=squats_graph_xdata,
                           l_reps=l_reps, r_reps=r_reps, rh_reps=rh_reps, lh_reps=lh_reps, rl_elbow=rl_elbow,
                           ll_elbow=ll_elbow, b_timestamp=b_timestamp)

@app.route('/login/', methods=['POST', 'GET'])
def Login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            cursor = db.cursor()
            cursor.execute('SELECT * FROM user WHERE username=%s', (username, ))
            user = cursor.fetchall()
            user = pd.DataFrame(user, columns=user_label)
            cursor.close()
            pas = user['password'][0]
            if sha256_crypt.verify(password, pas):
                session['username'] = username
                session['logged_in'] = True
                flash("Successfully loggedin")
                return redirect(url_for('index2'))
            else:
                flash("invalid password")
                return redirect(url_for('index2'))
        except:
            flash("invalid username")
            return redirect(url_for('index2'))
        
    return render_template('profile.html')

@app.route('/signin/', methods=['POST', 'GET'])
def Signin():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        encPass = sha256_crypt.encrypt(password)
        full_name = request.form['full_name']
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        gender = request.form['gender']
        cursor = db.cursor()
        
        try:
            cursor.execute('''INSERT INTO user(username, email, password, full_name, age, height, weight, gender)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)''', (username, email, encPass, full_name, age, height, weight, gender))
            db.commit()
            session['username'] = username
            session['logged_in'] = True
            cursor.close()
            return redirect(url_for('index2'))
        except:
            flash("Already exists")
            return redirect(url_for('index2'))

@app.route('/logout/')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('index2'))


mpDraw = mp.solutions.drawing_utils
my_pose = mp.solutions.pose
pose = my_pose.Pose()

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False

def calculateAngle(landmark1, landmark2, landmark3):



    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3


    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))


    if angle < 0:

        angle += 360


    return angle

# ind = 0
dem = 0
z = 1
# seconds_old = datetime.now().strftime("%S")
seconds_old = 0

global res, pos, frm, lst, pred


def count_time(time):
    global seconds_old, dem, z
    now = datetime.now()
    seconds_new = now.strftime("%S")
    if seconds_new != seconds_old:
        seconds_old = seconds_new
        dem = dem + 1
        if dem == time + 1 :
            dem = 0
            z += 1
            if z == 6:
                z = 1
    return dem, z

# Set the path to the folder containing the images
folder_path = 'yoga'

# Create a list to store the image paths
image_paths = []

# Loop through the files in the folder and add the paths to the list
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(folder_path, filename)
        image_paths.append(img_path)

ind = 0      
def next_image():
    global ind
    ind += 1
    if ind >= len(image_paths):
        ind = 0
    img = cv2.imread(image_paths[ind])
    return img
    # cv2.imshow("image", img)
    # Call the function again after a delay to switch to the next image
  
# folderPath = 'yoga'
# myList = os.listdir(folderPath)  # List of image
# overlayList = []  # list after read image
# for imgPath in myList:
#     image = cv2.imread(f'{folderPath}/{imgPath}')
#     overlayList.append(image)

def detectPose(image, pose):


    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    results = pose.process(imageRGB)

    height, width, _ = image.shape


    landmarks = []

    if results.pose_landmarks:

        mpDraw.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=my_pose.POSE_CONNECTIONS)


        for landmark in results.pose_landmarks.landmark:

            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              int(landmark.z * width)))

    return output_image, landmarks


def gen():
    model = load_model("model.h5")
    label = np.load("labels.npy")

    mpDraw = mp.solutions.drawing_utils
    my_pose = mp.solutions.pose
    pose = my_pose.Pose()

    cap = cv2.VideoCapture(0)
    
    dem = 0
    z = 1
    # seconds_old = datetime.now().strftime("%S")
    while True:
        lst = []
        x, y = 50, 50
        width, height = 200, 200
        _, frm = cap.read()


        # window = np.zeros((940, 940, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)
        image = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        # frm = cv2.resize(frm, (1280, 720))
        res = pose.process(image)
        frm , landmarks = detectPose(frm, pose)

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            if landmarks:
                left_elbow_angle = calculateAngle(landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value],
                                        landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value],
                                        landmarks[my_pose.PoseLandmark.LEFT_WRIST.value])

                # print("lft elbw",left_elbow_angle)

                right_elbow_angle = calculateAngle(landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[my_pose.PoseLandmark.RIGHT_WRIST.value])

                left_shoulder_angle = calculateAngle(landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value],
                                            landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value],
                                            landmarks[my_pose.PoseLandmark.LEFT_HIP.value])


                right_shoulder_angle = calculateAngle(landmarks[my_pose.PoseLandmark.RIGHT_HIP.value],
                                            landmarks[my_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                            landmarks[my_pose.PoseLandmark.RIGHT_ELBOW.value])


                left_knee_angle = calculateAngle(landmarks[my_pose.PoseLandmark.LEFT_HIP.value],
                                        landmarks[my_pose.PoseLandmark.LEFT_KNEE.value],
                                        landmarks[my_pose.PoseLandmark.LEFT_ANKLE.value])

                right_knee_angle = calculateAngle(landmarks[my_pose.PoseLandmark.RIGHT_HIP.value],
                                        landmarks[my_pose.PoseLandmark.RIGHT_KNEE.value],
                                        landmarks[my_pose.PoseLandmark.RIGHT_ANKLE.value])
            else:

                cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255),3,cv2.LINE_AA)

            # print("right knee = ",right_knee_angle)
            
        # frm = cv2.blur(frm, (4, 4))
        # if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]


            if p[0][np.argmax(p)] > 0.75:
                # cv2.rectangle(frm, (0, 0), (225, 73), (245, 117, 16), -1)
                # cv2.putText(frm, pred , (50, 50), cv2.FONT_ITALIC, 1.5, (0,255,0),2)
                if z == 1:
                    ind = 0
                    img = cv2.imread(image_paths[ind])
                    cv2.rectangle(frm, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    img_resized = cv2.resize(img, (width, height))
                    frm[y:y+height, x:x+width] = img_resized
                    # cv2.imshow("image", img)
                    if pred == "Pranamasana":
                        dem=0
                        dem, z = count_time(5)
                        print("function called")
                      
                        cv2.putText(frm, f"Pose held for {dem} seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                    
                        if  dem > 0:
                            
                                cv2.putText(frm, f"Pose held for {dem} seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                        elif dem == 0 and z == 1:    
                                cv2.putText(frm, "Hold the pose for 5 seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                        elif dem == 0 and z != 1:
                                cv2.putText(frm, "Next pose", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                        else: 
                            dem = 0        

                
                if z == 2:
                    ind= 1
                    img = cv2.imread(image_paths[ind])
                    cv2.rectangle(frm, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    img_resized = cv2.resize(img, (width, height))
                    frm[y:y+height, x:x+width] = img_resized
                    if pred == "viebhadrasana":
                        if left_elbow_angle > 165 and left_elbow_angle < 195 and right_elbow_angle > 165 and right_elbow_angle < 195:


                            if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 80 and right_shoulder_angle < 110:


                                 if (left_knee_angle > 165 and left_knee_angle < 195) or (right_knee_angle > 165 and right_knee_angle < 195):

                                    if (left_knee_angle > 90 and left_knee_angle < 120) or (right_knee_angle > 90 and right_knee_angle < 120):
                        
                                        dem, z = count_time(5)
                                        if dem > 0 or dem==0:
                                            cv2.putText(frm, f"Pose held for {dem} seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                                            if pred == "nothing":
                                                dem=0
                                        elif dem == 0 and z == 2:
                                            cv2.putText(frm, "Hold the pose for 5 seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                                        elif dem == 0 and z != 2:
                                            cv2.putText(frm, "Next pose", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                                            # next_image()
                                    # if pred == "nothing":
                                    #     dem=0
                                    else: 
                                        dem = 0

                  
                if z == 3:
                    ind= 2
                    img = cv2.imread(image_paths[ind])
                    cv2.rectangle(frm, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    img_resized = cv2.resize(img, (width, height))
                    frm[y:y+height, x:x+width] = img_resized
                    if pred == "vrikshasana" :
                        if (left_knee_angle > 165 and left_knee_angle < 195) or (right_knee_angle > 165 and right_knee_angle < 195):

                            if (left_knee_angle > 315 and left_knee_angle < 335) or (right_knee_angle > 25 and right_knee_angle < 45):
                                dem, z = count_time(5)
                                if dem > 0:
                                    cv2.putText(frm, f"Pose held for {dem} seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                                elif dem == 0 and z == 3:
                                    cv2.putText(frm, "Hold the pose for 5 seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                                elif dem == 0 and z != 3:
                                    cv2.putText(frm, "Next pose", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                            else:
                                    dem = 0           
                
                if z == 4:
                    ind = 3
                    img = cv2.imread(image_paths[ind])
                    cv2.rectangle(frm, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    img_resized = cv2.resize(img, (width, height))
                    frm[y:y+height, x:x+width] = img_resized
                    if pred == "parsvakonasana":
                        dem, z = count_time(5)
                        if dem > 0:
                            cv2.putText(frm, f"Pose held for {dem} seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                        elif dem == 0 and z == 4:
                            cv2.putText(frm, "Hold the pose for 5 seconds", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                        elif dem == 0 and z != 4:
                            cv2.putText(frm, "Next pose", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                    else:
                            dem = 0 

                if z == 5:
                    ind =4
                    img = cv2.imread(image_paths[ind])
                    cv2.rectangle(frm, (x, y), (x + width, y + height), (0, 0, 255), 2)
                    img_resized = cv2.resize(img, (width, height))
                    frm[y:y+height, x:x+width] = img_resized
                    cv2.rectangle(frm, (0, 450), (650, 650), (170, 232, 238), cv2.FILLED)
                    cv2.putText(frm, f"yoga done!", (10, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 15)

            else:
                dem = 0
            #     cv2.putText(frm, "Asana is either wrong or not trained" , (30, 100), cv2.FONT_ITALIC, 1.3, (0,0,255),3,cv2.LINE_AA)

        else:

            cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0,255),3,cv2.LINE_AA)

        mpDraw.draw_landmarks(frm, res.pose_landmarks, my_pose.POSE_CONNECTIONS,
                              mpDraw.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=2),
                              mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))

        cv2.imshow("window", frm)

        frame = cv2.imencode('.jpg', frm)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



def calculate_angle(point1: list, point2: list, point3: list) -> float:
    '''
    Calculate the angle between 3 points
    Unit of the angle will be in Degree
    '''
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)

    # Calculate algo
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    return angleInDeg

class BicepPoseAnalysis:
    def __init__(self, side: str, stage_down_threshold: float, stage_up_threshold: float, peak_contraction_threshold: float, loose_upper_arm_angle_threshold: float, visibility_threshold: float):
        # Initialize thresholds
        self.stage_down_threshold = stage_down_threshold
        self.stage_up_threshold = stage_up_threshold
        self.peak_contraction_threshold = peak_contraction_threshold
        self.loose_upper_arm_angle_threshold = loose_upper_arm_angle_threshold
        self.visibility_threshold = visibility_threshold

        self.side = side
        self.counter = 0
        self.stage = "down"
        self.is_visible = True
        self.detected_errors = {
            "LOOSE_ELBOW": 0,
            "HALF_REP": 0,
        }

        # Params for loose upper arm error detection
        self.loose_upper_arm = False

        # Params for peak contraction error detection
        self.peak_contraction_angle = 1000
        self.peak_contraction_frame = None
    
    def get_joints(self, landmarks) -> bool:
        '''
        Check for joints' visibility then get joints coordinate
        '''
        side = self.side.upper()

        # Check visibility
        joints_visibility = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].visibility, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].visibility ]

        is_visible = all([ vis > self.visibility_threshold for vis in joints_visibility ])
        self.is_visible = is_visible

        if not is_visible:
            return self.is_visible
        
        # Get joints' coordinates
        self.shoulder = [ landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_SHOULDER"].value].y ]
        self.elbow = [ landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_ELBOW"].value].y ]
        self.wrist = [ landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].x, landmarks[mp_pose.PoseLandmark[f"{side}_WRIST"].value].y ]

        return self.is_visible
    
    def analyze_pose(self,results, landmarks, frame):
        '''
        - Bicep Counter
        - Errors Detection
        '''
        self.get_joints(landmarks)

        # Cancel calculation if visibility is poor
        if not self.is_visible:
            return (None, None)

        # * Calculate curl angle for counter
        bicep_curl_angle = int(calculate_angle(self.shoulder, self.elbow, self.wrist))
        if bicep_curl_angle > self.stage_down_threshold:
            self.stage = "down"
        elif bicep_curl_angle < self.stage_up_threshold and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            count = self.counter
        
        # * Calculate the angle between the upper arm (shoulder & joint) and the Y axis
        shoulder_projection = [ self.shoulder[0], 1 ] # Represent the projection of the shoulder to the X axis
        ground_upper_arm_angle = int(calculate_angle(self.elbow, self.shoulder, shoulder_projection))

        # * Evaluation for LOOSE UPPER ARM error
        if ground_upper_arm_angle > self.loose_upper_arm_angle_threshold:
            # Limit the saved frame
            if not self.loose_upper_arm:
                self.loose_upper_arm = True
                # save_frame_as_image(frame, f"Loose upper arm: {ground_upper_arm_angle}")
                self.detected_errors["LOOSE_ELBOW"] += 1
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))
        else:
            self.loose_upper_arm = False
        
        # * Evaluate PEAK CONTRACTION error
        if self.stage == "up" and bicep_curl_angle < self.peak_contraction_angle:
            # Save peaked contraction every rep
            self.peak_contraction_angle = bicep_curl_angle
            self.peak_contraction_frame = frame
            
        elif self.stage == "down":
            # * Evaluate if the peak is higher than the threshold if True, marked as an error then saved that frame
            if self.peak_contraction_angle != 1000 and self.peak_contraction_angle >= self.peak_contraction_threshold:
                # save_frame_as_image(self.peak_contraction_frame, f"{self.side} - Peak Contraction: {self.peak_contraction_angle}")
                self.detected_errors["HALF_REP"] += 1
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))

            # Reset params
            self.peak_contraction_angle = 1000
            self.peak_contraction_frame = None
        
        return (bicep_curl_angle, ground_upper_arm_angle)


def gen1(username, start_timestamp):
    cap = cv2.VideoCapture(0)

    VISIBILITY_THRESHOLD = 0.65

# Params for counter
    STAGE_UP_THRESHOLD = 90
    STAGE_DOWN_THRESHOLD = 120

# Params to catch FULL RANGE OF MOTION error
    PEAK_CONTRACTION_THRESHOLD = 60

# LOOSE UPPER ARM error detection
    LOOSE_UPPER_ARM = False
    LOOSE_UPPER_ARM_ANGLE_THRESHOLD = 40

# STANDING POSTURE error detection
# POSTURE_ERROR_THRESHOLD = 0.7
# posture = "C" 

# Init analysis class
    left_arm_analysis = BicepPoseAnalysis(side="left", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

    right_arm_analysis = BicepPoseAnalysis(side="right", stage_down_threshold=STAGE_DOWN_THRESHOLD, stage_up_threshold=STAGE_UP_THRESHOLD, peak_contraction_threshold=PEAK_CONTRACTION_THRESHOLD, loose_upper_arm_angle_threshold=LOOSE_UPPER_ARM_ANGLE_THRESHOLD, visibility_threshold=VISIBILITY_THRESHOLD)

    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
        while cap.isOpened():
            ret, image = cap.read()

            if not ret:
                break

            # Reduce size of a frame
            # image = rescale_frame(image, 50)
            # image = cv2.flip(image, 1)
            
            video_dimensions = [image.shape[1], image.shape[0]]

            # Recolor image from BGR to RGB for mediapipe
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            if not results.pose_landmarks:
                # print("No human found")
                continue

            # Recolor image from BGR to RGB for mediapipe
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))

            # Make detection
            try:
                landmarks = results.pose_landmarks.landmark
                
                (left_bicep_curl_angle, left_ground_upper_arm_angle) = left_arm_analysis.analyze_pose(results,landmarks=landmarks, frame=image)
                (right_bicep_curl_angle, right_ground_upper_arm_angle) = right_arm_analysis.analyze_pose(results,landmarks=landmarks, frame=image)        
                

                cv2.rectangle(image, (0, 0), (500, 40), (245, 117, 16), -1)

                # Display probability
                cv2.putText(image, "RIGHT", (15, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.counter) if right_arm_analysis.is_visible else "UNK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Display Left Counter
                cv2.putText(image, "LEFT", (95, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.counter) if left_arm_analysis.is_visible else "UNK", (100, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # * Display error
                # Right arm error
                cv2.putText(image, "R_Half", (165, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.detected_errors["HALF_REP"]), (160, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "R_Loose", (225, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(right_arm_analysis.detected_errors["LOOSE_ELBOW"]), (220, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Left arm error
                cv2.putText(image, "L_Half", (300, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.detected_errors["HALF_REP"]), (295, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, "L_Loose", (380, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(left_arm_analysis.detected_errors["LOOSE_ELBOW"]), (375, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Lean back error
                # cv2.putText(image, "LB", (460, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                # cv2.putText(image, str(f"{posture}, {predicted_class}, {class_prediction_probability}"), (440, 30), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


                # * Visualize angles
                # Visualize LEFT arm calculated angles
                if left_arm_analysis.is_visible:
                    cv2.putText(image, str(left_bicep_curl_angle), tuple(np.multiply(left_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(left_ground_upper_arm_angle), tuple(np.multiply(left_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


                # Visualize RIGHT arm calculated angles
                if right_arm_analysis.is_visible:
                    cv2.putText(image, str(right_bicep_curl_angle), tuple(np.multiply(right_arm_analysis.elbow, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(right_ground_upper_arm_angle), tuple(np.multiply(right_arm_analysis.shoulder, video_dimensions).astype(int)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            except Exception as e:
                print(f"Error: {e}")

            counter_R =  right_arm_analysis.counter
            counter_L =  left_arm_analysis.counter
            r_peak = right_arm_analysis.detected_errors["HALF_REP"]
            l_peak = left_arm_analysis.detected_errors["HALF_REP"]
            r_lua = right_arm_analysis.detected_errors["LOOSE_ELBOW"]
            l_lua = left_arm_analysis.detected_errors["LOOSE_ELBOW"]
            if counter_R >= 5 and counter_L >= 5:
                cursor = db.cursor()
                cursor.execute('''INSERT INTO Biceprepdata (exer_name, l_reps, r_reps, lh_reps, rh_reps, ll_relbow, rl_elbow, timestamp, username, start_timestamp)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''', ('Bicep Curl', counter_L, counter_R, l_peak, r_peak, l_lua, r_lua, datetime.now(), username, start_timestamp))
                db.commit()
                cursor.close()
                break

            # if counter_R >=5 and counter_L >=5:
            #     break
            cv2.imshow("CV2", image)
        
            
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            #        b'--frame\r\n'
            #                b'Content-Type: application/json\r\n\r\n' + json_data + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break

@app.route('/video_feed1')
def video_feed1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    username = session['username']
    start_timestamp = datetime.now()
    return Response(gen1(username, start_timestamp),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def calculate_angle(point1: list, point2: list, point3: list) -> float:
    point1 = np.array(point1)
    point2 = np.array(point2)
    point3 = np.array(point3)
    angleInRad = np.arctan2(point3[1] - point2[1], point3[0] - point2[0]) - np.arctan2(point1[1] - point2[1], point1[0] - point2[0])
    angleInDeg = np.abs(angleInRad * 180.0 / np.pi)

    angleInDeg = angleInDeg if angleInDeg <= 180 else 360 - angleInDeg
    
    return angleInDeg




def rescale_frame(frame, percent=50):
    '''
    Rescale a frame to a certain percentage compare to its original frame
    '''
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def calculate_distance(pointX, pointY) -> float:
    '''
    Calculate a distance between 2 points
    '''

    x1, y1 = pointX
    x2, y2 = pointY

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def analyze_foot_knee_placement(results, stage: str, foot_shoulder_ratio_thresholds: list, knee_foot_ratio_thresholds: dict, visibility_threshold: int) -> dict:
    '''
    Calculate the ratio between the foot and shoulder for FOOT PLACEMENT analysis
    
    Calculate the ratio between the knee and foot for KNEE PLACEMENT analysis

    Return result explanation:
        -1: Unknown result due to poor visibility
        0: Correct knee placement
        1: Placement too tight
        2: Placement too wide
    '''
    analyzed_results = {
        "foot_placement": -1,
        "knee_placement": -1,
    }

    landmarks = results.pose_landmarks.landmark

    # * Visibility check of important landmarks for foot placement analysis
    left_foot_index_vis = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility
    right_foot_index_vis = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility

    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_foot_index_vis < visibility_threshold or right_foot_index_vis < visibility_threshold or left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        return analyzed_results
    
    # * Calculate shoulder width
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    # * Calculate 2-foot width
    left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
    right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    foot_width = calculate_distance(left_foot_index, right_foot_index)
    # print("foot_width",foot_width)
    # * Calculate foot and shoulder ratio
    foot_shoulder_ratio = round(foot_width / shoulder_width, 1)

    # * Analyze FOOT PLACEMENT
    min_ratio_foot_shoulder, max_ratio_foot_shoulder = foot_shoulder_ratio_thresholds
    if min_ratio_foot_shoulder <= foot_shoulder_ratio <= max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 0
    elif foot_shoulder_ratio < min_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 1
    elif foot_shoulder_ratio > max_ratio_foot_shoulder:
        analyzed_results["foot_placement"] = 2
    
    # * Visibility check of important landmarks for knee placement analysis
    left_knee_vis = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    right_knee_vis = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility

    # If visibility of any keypoints is low cancel the analysis
    if (left_knee_vis < visibility_threshold or right_knee_vis < visibility_threshold):
        print("Cannot see foot")
        return analyzed_results

    # * Calculate 2 knee width
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    knee_width = calculate_distance(left_knee, right_knee)
    # print("knee width",knee_width)
    # * Calculate foot and shoulder ratio
    knee_foot_ratio = round(knee_width / foot_width, 1)

    # * Analyze KNEE placement
    up_min_ratio_knee_foot, up_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("up")
    middle_min_ratio_knee_foot, middle_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("middle")
    down_min_ratio_knee_foot, down_max_ratio_knee_foot = knee_foot_ratio_thresholds.get("down")

    if stage == "up":
        if up_min_ratio_knee_foot <= knee_foot_ratio <= up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < up_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > up_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "middle":
        if middle_min_ratio_knee_foot <= knee_foot_ratio <= middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < middle_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > middle_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    elif stage == "down":
        if down_min_ratio_knee_foot <= knee_foot_ratio <= down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 0
        elif knee_foot_ratio < down_min_ratio_knee_foot:
            analyzed_results["knee_placement"] = 1
        elif knee_foot_ratio > down_max_ratio_knee_foot:
            analyzed_results["knee_placement"] = 2
    
    return analyzed_results

def gen2(username, start_timestamp):
    cap = cv2.VideoCapture(0)
    # global incorrect_f, incorrect_k
# Counter vars
    counter = 0
    current_stage = ""
    PREDICTION_PROB_THRESHOLD = 0.7

    # Error vars
    VISIBILITY_THRESHOLD = 0.6
    FOOT_SHOULDER_RATIO_THRESHOLDS = [1.2, 2.8]
    KNEE_FOOT_RATIO_THRESHOLDS = {
        "up": [0.5, 1.0],
        "middle": [0.8, 1.0],
        "down": [0.8, 1.1],
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
     while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break
        
        # Reduce size of a frame
        image = rescale_frame(image, 100)

        # Recolor image from BGR to RGB for mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)
        if not results.pose_landmarks:
            continue

        # Recolor image from BGR to RGB for mediapipe
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw landmarks and connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(244, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1))
        # mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))

        # Make detection
        try:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            
            
            angle_knee = calculate_angle(hip, left_knee, ankle) #Knee joint angle
            
            angle_hip = calculate_angle(left_shoulder, hip, left_knee)
            hip_angle = 180-angle_hip
            knee_angle = 180-angle_knee
            if angle_knee > 169:
                current_stage = "up"
            if angle_knee <=169 and angle_knee>90:
                current_stage=="middle"    
            if angle_knee <= 90 and current_stage =='up':
                current_stage="down"
                counter +=1
            

            # Analyze squat pose
            analyzed_results = analyze_foot_knee_placement(results=results, stage=current_stage, foot_shoulder_ratio_thresholds=FOOT_SHOULDER_RATIO_THRESHOLDS, knee_foot_ratio_thresholds=KNEE_FOOT_RATIO_THRESHOLDS, visibility_threshold=VISIBILITY_THRESHOLD)

            foot_placement_evaluation = analyzed_results["foot_placement"]
            knee_placement_evaluation = analyzed_results["knee_placement"]
            
            

            # * Evaluate FOOT PLACEMENT error
            if foot_placement_evaluation == -1:
                foot_placement = "UNK"
            elif foot_placement_evaluation == 0:
                foot_placement = "Correct"

            elif foot_placement_evaluation == 1:
                foot_placement = "Too tight"
                # incorrect_f=incorrect_f+1
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))

            elif foot_placement_evaluation == 2:
                foot_placement = "Too wide"
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))
                # incorrect_f=incorrect_f+1
            
            # * Evaluate KNEE PLACEMENT error
            if knee_placement_evaluation == -1:
                knee_placement = "UNK"

            elif knee_placement_evaluation == 0:
                knee_placement = "Correct"

            elif knee_placement_evaluation == 1:
                knee_placement = "Too tight"
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))
                # incorrect_k=incorrect_k+1

            elif knee_placement_evaluation == 2:
                knee_placement = "Too wide"
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(0,100,255), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(100,100,255), thickness=2, circle_radius=1))
                # incorrect_k=incorrect_k+1

            # Visualization
            # Status box
            cv2.rectangle(image, (0, 0), (500, 60), (245, 117, 16), -1)

            # Display class
            cv2.putText(image, "COUNT", (10, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter) , (5, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

            # Display Foot and Shoulder width ratio
            cv2.putText(image, "FOOT", (200, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, foot_placement, (195, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)

            # Display knee and Shoulder width ratio
            cv2.putText(image, "KNEE", (330, 12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, knee_placement, (325, 40), cv2.FONT_HERSHEY_COMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)


        except Exception as e:
            print(f"Error: {e}")
        
        if counter==5:
            cursor = db.cursor()
            cursor.execute('''INSERT INTO Squatsrepdata (exer_name, rep, timestamp, username, start_timestamp)
                        VALUES (%s, %s, %s, %s, %s)''', ('Squats', counter, datetime.now(), username, start_timestamp))
            db.commit()
            cursor.close()
            break

        cv2.imshow("CV2", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        key = cv2.waitKey(20)
        if key == 27:
           break


@app.route('/video_feed2')
def video_feed2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    username = session['username']
    start_timestamp = datetime.now()
    return Response(gen2(username, start_timestamp),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True, port="8080")

