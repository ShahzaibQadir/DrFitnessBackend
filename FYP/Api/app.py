import pandas as pd
import cv2
import mediapipe as mp
import numpy as np
from math import acos,degrees
from flask_restful import Resource,Api
from flask import Flask
from math import acos,degrees 
from logging import FileHandler,WARNING

app = Flask(__name__)
api=Api(app)
#squats 
file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)
#homepage
# user_path="C:\\Users\\admin\\Desktop\\Kaggle\\Dataset result\\gender_submission.csv"
class Squats(Resource):
    def get(self):
        
        mp_drawing=mp.solutions.drawing_utils
        mp_pose=mp.solutions.pose
        cap=cv2.VideoCapture(0)

        up=False
        down=False
        count=0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                if ret == False:
                    break
                frame=cv2.flip(frame,1)
                height,width,_=frame.shape
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    x1=int(results.pose_landmarks.landmark[24].x*width)
                    y1=int(results.pose_landmarks.landmark[24].y*height)
                    
                    x2=int(results.pose_landmarks.landmark[26].x*width)
                    y2=int(results.pose_landmarks.landmark[26].y*height)
                    
                    x3=int(results.pose_landmarks.landmark[28].x*width)
                    y3=int(results.pose_landmarks.landmark[28].y*height)
                    
                    p1=np.array([x1,y1])
                    p2=np.array([x2,y2])
                    p3=np.array([x3,y3])
                    
                    l1=np.linalg.norm(p2-p3)
                    l2=np.linalg.norm(p1-p3)
                    l3=np.linalg.norm(p1-p2)
                    
                    #calculate angle
                    angle=degrees(acos((l1**2 + l3**2 - l2**2)/ (2* l1*l3)))
                    if angle>=160:
                        up=True
                    if up==True and down ==False and angle<=110:
                        down=True
                    if up==True and down ==True and angle>=160:
                        count+=1
                        up=False
                        down=False
                    print("count",count)
                        
                    #Visualization
                    aux_image=np.zeros(frame.shape, np.uint8)
        #             cv2.line(aux_image, (x1,y1),(x2,y2),(255,255,0),20)
        #             cv2.line(aux_image, (x2,y2),(x3,y3),(255,255,0),20)
        #             cv2.line(aux_image, (x1,y1),(x3,y3),(255,255,0),5)
                    
        #             contours=np.array([[x1,y1],[x2,y2],[x3,y3]])
        #             cv2.fillPoly(aux_image,pts=[contours],color=(128,0,250))
                    
                    output=cv2.addWeighted(frame,1,aux_image,0.8,0)
                    
                    
        #             cv2.circle(output, (x1,y1),6,(0,255,255),4)
        #             cv2.circle(output, (x3,y3),6,(128,0,250),4)
        #             cv2.circle(output, (x3,y3),6,(255,191,0),4)
                    cv2.rectangle(output,(2,1),(100,60),(25,25,0),-1)
                    cv2.putText(output,str(int(angle)),(x2 +30,y2),1,1.5,(0,0,255),2)
                    cv2.putText(output,str(count),(10,50),1,3.5,(0,0,255),2)
                    
                    

                    #cv2.imshow("aux_image",output)
                    
                    cv2.imshow("(Dr.Fitness) Correct Squats Counter",output)
                if cv2.waitKey(1) & 0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
        # data=pd.read_csv(user_path)
        # data=data.to_dict()
        # return {'data':data},200
     
class Shoulder(Resource):
    def get(self):
        
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        def findPosition(image, draw=True):
            lmList = []
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
            return lmList
        
        counter=0
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
        
                if ret == False:
                    break
        
                frame = cv2.flip(frame,1)
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    if results.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                        lmList = findPosition(image, draw=True)
            #             if len(lmList) != 0:
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
                        if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[14][1], lmList[14][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[13][1], lmList[13][2]), 20, (0, 255, 0), cv2.FILLED)
                            stage = "down"
                        if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":
                            stage = "up"
                            counter += 1
                            print(counter)

                except:
                    pass
                text = "{}:{}".format("Shoulder ", counter)
                cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
                cv2.imshow('Shoulder Counter ', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
    
class Bicep_Updated_Right_Hand(Resource):
    def get(self):
        #Right Hand
                
        mp_drawing=mp.solutions.drawing_utils
        mp_pose=mp.solutions.pose
        cap=cv2.VideoCapture(0)
        up=False
        down=False
        count=0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                if ret == False:
                    break
                #frame=cv2.flip(frame,1)
                height,width,_=frame.shape
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    x1=int(results.pose_landmarks.landmark[12].x*width)
                    y1=int(results.pose_landmarks.landmark[12].y*height)
                    
                    x2=int(results.pose_landmarks.landmark[14].x*width)
                    y2=int(results.pose_landmarks.landmark[14].y*height)
                    
                    x3=int(results.pose_landmarks.landmark[16].x*width)
                    y3=int(results.pose_landmarks.landmark[16].y*height)
                    
                    p1=np.array([x1,y1])
                    p2=np.array([x2,y2])
                    p3=np.array([x3,y3])
                    
                    l1=np.linalg.norm(p2-p3)
                    l2=np.linalg.norm(p1-p3)
                    l3=np.linalg.norm(p1-p2)
                    
                    #calculate angle
                    angle=degrees(acos((l1**2 + l3**2 - l2**2)/ (2* l1*l3)))
                    if angle>=140:
                        up=False
                        down=True
                    if angle<110:
                        down=False
                        up=True
                    if up==True and down==False and angle>=110:
                        count+=1
                        down=True
                        
                    print("count",count)
                        
                    #Visualization
                    aux_image=np.zeros(frame.shape, np.uint8)
        #             cv2.line(aux_image, (x1,y1),(x2,y2),(255,255,0),20)
        #             cv2.line(aux_image, (x2,y2),(x3,y3),(255,255,0),20)
        #             cv2.line(aux_image, (x1,y1),(x3,y3),(255,255,0),5)
                    
        #             contours=np.array([[x1,y1],[x2,y2],[x3,y3]])
        #             cv2.fillPoly(aux_image,pts=[contours],color=(128,0,250))
                    
                    output=cv2.addWeighted(frame,1,aux_image,0.8,0)
                    
                    
        #             cv2.circle(output, (x1,y1),6,(0,255,255),4)
        #             cv2.circle(output, (x3,y3),6,(128,0,250),4)
        #             cv2.circle(output, (x3,y3),6,(255,191,0),4)
                    cv2.rectangle(output,(2,1),(100,60),(25,25,0),-1)
                    cv2.putText(output,str(int(angle)),(x2 +30,y2),1,1.5,(0,0,255),2)
                    cv2.putText(output,str(count),(10,50),1,3.5,(0,0,255),2)
                    
                    

                    #cv2.imshow("aux_image",output)
                    
                    cv2.imshow("(Dr.Fitness) Correct Bicep Counter",output)
                if cv2.waitKey(1) & 0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()

class Bicep_Updated_Left_Hand(Resource):
    def get(self):
        #Left Hand

        mp_drawing=mp.solutions.drawing_utils
        mp_pose=mp.solutions.pose
        cap=cv2.VideoCapture(0)
        up=False
        down=False
        count=0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                if ret == False:
                    break
                frame=cv2.flip(frame,1)
                height,width,_=frame.shape
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    x1=int(results.pose_landmarks.landmark[11].x*width)
                    y1=int(results.pose_landmarks.landmark[11].y*height)
                    
                    x2=int(results.pose_landmarks.landmark[13].x*width)
                    y2=int(results.pose_landmarks.landmark[13].y*height)
                    
                    x3=int(results.pose_landmarks.landmark[15].x*width)
                    y3=int(results.pose_landmarks.landmark[15].y*height)
                    
                    p1=np.array([x1,y1])
                    p2=np.array([x2,y2])
                    p3=np.array([x3,y3])
                    
                    l1=np.linalg.norm(p2-p3)
                    l2=np.linalg.norm(p1-p3)
                    l3=np.linalg.norm(p1-p2)
                    
                    #calculate angle
                    angle=degrees(acos((l1**2 + l3**2 - l2**2)/ (2* l1*l3)))
                    if angle>=140:
                        up=False
                        down=True
                    if angle<110:
                        down=False
                        up=True
                    if up==True and down==False and angle>=110:
                        count+=1
                        down=True
                        
                    print("count",count)
                        
                    #Visualization
                    aux_image=np.zeros(frame.shape, np.uint8)
        #             cv2.line(aux_image, (x1,y1),(x2,y2),(255,255,0),20)
        #             cv2.line(aux_image, (x2,y2),(x3,y3),(255,255,0),20)
        #             cv2.line(aux_image, (x1,y1),(x3,y3),(255,255,0),5)
                    
        #             contours=np.array([[x1,y1],[x2,y2],[x3,y3]])
        #             cv2.fillPoly(aux_image,pts=[contours],color=(128,0,250))
                    
                    output=cv2.addWeighted(frame,1,aux_image,0.8,0)
                    
                    
        #             cv2.circle(output, (x1,y1),6,(0,255,255),4)
        #             cv2.circle(output, (x3,y3),6,(128,0,250),4)
        #             cv2.circle(output, (x3,y3),6,(255,191,0),4)
                    cv2.rectangle(output,(2,1),(100,60),(25,25,0),-1)
                    cv2.putText(output,str(int(angle)),(x2 +30,y2),1,1.5,(0,0,255),2)
                    cv2.putText(output,str(count),(10,50),1,3.5,(0,0,255),2)
                    
                    

                    #cv2.imshow("aux_image",output)
                    
                    cv2.imshow("(Dr.Fitness) Correct Bicep Counter",output)
                if cv2.waitKey(1) & 0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
class Push_Ups(Resource):
    def get(self):
        
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        def findPosition(image, draw=True):
            lmList = []
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
            return lmList
        
        counter=0
        cap = cv2.VideoCapture(0)

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
        
                if ret == False:
                    break
        
                frame = cv2.flip(frame,1)
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
            
                # Make detection
                results = pose.process(image)
            
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                try:
                    if results.pose_landmarks is not None:
                        mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                        lmList = findPosition(image, draw=True)
            #             if len(lmList) != 0:
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
                        if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
            #                 cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[14][1], lmList[14][2]), 20, (0, 255, 0), cv2.FILLED)
            #                 cv2.circle(image, (lmList[13][1], lmList[13][2]), 20, (0, 255, 0), cv2.FILLED)
                            stage = "down"
                        if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "down":
                            stage = "up"
                            counter += 1
                            print(counter)

                except:
                    pass
                text = "{}:{}".format("Push Ups", counter)
                cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2)
                cv2.imshow('Push Ups Counter', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

class Tricep_Right_Hand(Resource):
    def get(self):
        mp_drawing=mp.solutions.drawing_utils
        mp_pose=mp.solutions.pose
        cap=cv2.VideoCapture(0)
        down=False
        count=0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                if ret == False:
                    break
                frame=cv2.flip(frame,1)
                height,width,_=frame.shape
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    x1=int(results.pose_landmarks.landmark[12].x*width)
                    y1=int(results.pose_landmarks.landmark[12].y*height)
                    
                    x2=int(results.pose_landmarks.landmark[14].x*width)
                    y2=int(results.pose_landmarks.landmark[14].y*height)
                    
                    x3=int(results.pose_landmarks.landmark[16].x*width)
                    y3=int(results.pose_landmarks.landmark[16].y*height)
                    
                    p1=np.array([x1,y1])
                    p2=np.array([x2,y2])
                    p3=np.array([x3,y3])
                    
                    l1=np.linalg.norm(p2-p3)
                    l2=np.linalg.norm(p1-p3)
                    l3=np.linalg.norm(p1-p2)
                    
                    #calculate angle
                    angle=degrees(acos((l1**2 + l3**2 - l2**2)/ (2* l1*l3)))
                    if angle>=140:
                        up=False
                        down=True
                    if angle<110:
                        down=False
                        up=True
                    if up==True and down==False and angle>=110:
                        count+=1
                        down=True
                        
                    print("count",count)
                        
                    #Visualization
                    aux_image=np.zeros(frame.shape, np.uint8)
        #             cv2.line(aux_image, (x1,y1),(x2,y2),(255,255,0),20)
        #             cv2.line(aux_image, (x2,y2),(x3,y3),(255,255,0),20)
        #             cv2.line(aux_image, (x1,y1),(x3,y3),(255,255,0),5)
                    
        #             contours=np.array([[x1,y1],[x2,y2],[x3,y3]])
        #             cv2.fillPoly(aux_image,pts=[contours],color=(128,0,250))
                    
                    output=cv2.addWeighted(frame,1,aux_image,0.8,0)
                    
                    
        #             cv2.circle(output, (x1,y1),6,(0,255,255),4)
        #             cv2.circle(output, (x3,y3),6,(128,0,250),4)
        #             cv2.circle(output, (x3,y3),6,(255,191,0),4)
                    cv2.rectangle(output,(2,1),(100,60),(25,25,0),-1)
                    cv2.putText(output,str(int(angle)),(x2 +30,y2),1,1.5,(0,0,255),2)
                    cv2.putText(output,str(count),(10,50),1,3.5,(0,0,255),2)
                    
                    

                    #cv2.imshow("aux_image",output)
                    
                    cv2.imshow("(Dr.Fitness) Correct Tricep Counter",output)
                if cv2.waitKey(1) & 0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
class Tricep_Left_Hand(Resource):
    def get(self):
        mp_drawing=mp.solutions.drawing_utils
        mp_pose=mp.solutions.pose
        cap=cv2.VideoCapture(0)
        up=False
        down=False
        count=0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret,frame=cap.read()
                if ret == False:
                    break
                #frame=cv2.flip(frame,1)
                height,width,_=frame.shape
                frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=pose.process(frame_rgb)
                
                if results.pose_landmarks is not None:
                    x1=int(results.pose_landmarks.landmark[11].x*width)
                    y1=int(results.pose_landmarks.landmark[11].y*height)
                    
                    x2=int(results.pose_landmarks.landmark[13].x*width)
                    y2=int(results.pose_landmarks.landmark[13].y*height)
                    
                    x3=int(results.pose_landmarks.landmark[15].x*width)
                    y3=int(results.pose_landmarks.landmark[15].y*height)
                    
                    p1=np.array([x1,y1])
                    p2=np.array([x2,y2])
                    p3=np.array([x3,y3])
                    
                    l1=np.linalg.norm(p2-p3)
                    l2=np.linalg.norm(p1-p3)
                    l3=np.linalg.norm(p1-p2)
                    
                    #calculate angle
                    angle=degrees(acos((l1**2 + l3**2 - l2**2)/ (2* l1*l3)))
                    if angle>=140:
                        up=False
                        down=True
                    if angle<110:
                        down=False
                        up=True
                    if up==True and down==False and angle>=110:
                        count+=1
                        down=True
                        
                    print("count",count)
                        
                    #Visualization
                    aux_image=np.zeros(frame.shape, np.uint8)
        #             cv2.line(aux_image, (x1,y1),(x2,y2),(255,255,0),20)
        #             cv2.line(aux_image, (x2,y2),(x3,y3),(255,255,0),20)
        #             cv2.line(aux_image, (x1,y1),(x3,y3),(255,255,0),5)
                    
        #             contours=np.array([[x1,y1],[x2,y2],[x3,y3]])
        #             cv2.fillPoly(aux_image,pts=[contours],color=(128,0,250))
                    
                    output=cv2.addWeighted(frame,1,aux_image,0.8,0)
                    
                    
        #             cv2.circle(output, (x1,y1),6,(0,255,255),4)
        #             cv2.circle(output, (x3,y3),6,(128,0,250),4)
        #             cv2.circle(output, (x3,y3),6,(255,191,0),4)
                    cv2.rectangle(output,(2,1),(100,60),(25,25,0),-1)
                    cv2.putText(output,str(int(angle)),(x2 +30,y2),1,1.5,(0,0,255),2)
                    cv2.putText(output,str(count),(10,50),1,3.5,(0,0,255),2)
                    
                    

                    #cv2.imshow("aux_image",output)
                    
                    cv2.imshow("(Dr.Fitness) Correct Tricep Counter",output)
                if cv2.waitKey(1) & 0xFF==27:
                    break
        cap.release()
        cv2.destroyAllWindows()
#api.com/users
api.add_resource(Squats,'/squats')
api.add_resource(Shoulder,'/shoulder')
api.add_resource(Bicep_Updated_Left_Hand,'/biceplefthand')
api.add_resource(Bicep_Updated_Right_Hand,'/biceprighthand')
api.add_resource(Push_Ups,'/pushup')
api.add_resource(Tricep_Right_Hand,'/triceprighthand')
api.add_resource(Tricep_Left_Hand,'/triceplefthand')



if __name__ == "__main__":
    app.run(debug=True)
    