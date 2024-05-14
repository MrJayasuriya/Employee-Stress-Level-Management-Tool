from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from .facerec.faster_video_stream import stream
from .facerec.click_photos import click
from .facerec.train_faces import trainer
from .models import Employee, Detected
from .forms import EmployeeForm
from face_rec_django.settings import MEDIA_ROOT
import cv2
import pickle
import face_recognition
import datetime
from cachetools import TTLCache
from sklearn.metrics.pairwise import euclidean_distances
import joblib
from joblib import load
from deepface import DeepFace
import threading
import threading
from collections import Counter
import joblib
import serial
import time
from django.core.mail import send_mail
from django.shortcuts import render
from .models import Employee
from django.contrib.auth.models import User
import csv


# ser = serial.Serial()
# ser.port = 'COM12'
# ser.baudrate = 9600
# ser.bytesize = 8
# ser.parity = serial.PARITY_NONE
# ser.stopbits = serial.STOPBITS_ONE

# def serialget():
#     value=[]
#     ser.open()
#     v=b'A'
#     ser.write(v)
#     while True:
#         for line in ser.read():
#             if chr(line) != '$':
#                 value.append(chr(line))
#             else:
#                 print("end")
#                 ser.close()
#                 return value


# def request(request):
    
#     str1=''
#     val=[]
#     va=serialget()
#     print(va)
#     for v in va:
#         if(v=='*'):
#             continue
#         else:  
#             if(v!='#'): 
#                 str1+=v
#             else:
#                 print(str1)
#                 val.append(float(str1))
#                 str1=""   
#     return render(request, 'app/add_emp.html', {'val1':int(val[0])})

cache = TTLCache(maxsize=20, ttl=60)

def save_data_continuously(name, emotion):
    # Implement your Django model saving logic here
    try:
        data = Employee.objects.get(name=name)
        data.emotion = emotion
        data.save()
    except Employee.DoesNotExist:
        print(f"No record found for {name}")

def daily_records(request):
    data = Employee.objects.all()
    return render(request, 'app/daily_record.html', {'models':data})

def show_database(request):
    data = Employee.objects.all()
    return render(request, 'app/show_database.html', {'models': data})
import os

def identify1(frame, name, buf, buf_length, known_conf):

    if name in cache:
        return
    count = 0
    for ele in buf:
        count += ele.count(name)
    
    if count >= known_conf:
        timestamp = datetime.now(tz=timezone.utc)
        print(name, timestamp)
        cache[name] = 'detected'
        path = 'G:/JAYASURYA/NEW OWN IN 2023/PROBLEMS/SHREYANTH DAS/FINAL REVIEW/FINAL REVIEW/Deploy/media/{}_{}.jpg'.format(name, timestamp)
       
        write_path = os.path.join(MEDIA_ROOT, path)
        cv2.imwrite(write_path, frame)
        try:
            emp = Employee.objects.get(name=name)
            emp.detected_set.create(time_stamp=timestamp, photo=path)
        except:
            pass 	        

def predict(rgb_frame, knn_clf=None, model_path=None, distance_threshold=0.5):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = joblib.load(f)

    # Load image file and find face locations
    # X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=2)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # print(closest_distances)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def get_user_id_by_name(name):
    try:
        user = Employee.objects.get(name=name)
        return user.id
    except User.DoesNotExist:
        return None

import csv

def identify_faces(video_capture):
    
    buf_length = 10
    known_conf = 6
    buf = [[]] * buf_length
    i = 0
    s1 =0
    # employee = Employee.objects.get(id=user_id)
    # if employee:
    #     name = employee.name
    #     emotion = employee.emotion
    #     heartrate = employee.heart_rate
        
    emotion_count = 0
    

    process_this_frame = True

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        if process_this_frame:
            predictions = predict(rgb_frame, model_path="G:/JAYASURYA/NEW OWN IN 2023/PROBLEMS/SHREYANTH DAS/FINAL REVIEW/FINAL REVIEW/Deploy/app/facerec/models/trained_model.clf")
            # result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            detected_emotion = result[0]['dominant_emotion']
            print(detected_emotion)

            # threading.Thread(target=save_data_continuously, args=(name, emotion)).start()
            # print(result[0]['dominant_emotion'])
            # emotion = result[0]['dominant_emotion']
            print("This is Dominant Emotion", detected_emotion)
            
            # emotion_count += 1
            # if emotion_count >= 30:
            #     print("The Emotion Limit Reached More than 30 sec")
            #     # threading.Thread(target=save_data_continuously, args=(name, emotion)).start()
            #     print('The data will Store the Countiuesly for specific User')
            #     emotions_list = []
                
            #     for _ in range(30):
            #         emotions_list.append(result[0]['dominant_emotion'])
            #         print("This is the emotion list", emotions_list)

            #     # Determine the dominant emotion based on counts
            #     dominant_emotion = Counter(emotions_list).most_common(1)[0][0]
            #     print("This is my dominant Emotion", dominant_emotion)
                
                # if dominant_emotion == 'happy'

                # Save dominant emotion in the database
                # employee, created = Employee.objects.get_or_create(name=name)
                # employee.dominant_emotion = dominant_emotion
                # employee.save()

        process_this_frame = not process_this_frame

        face_names = []

        for name, (top, right, bottom, left) in predictions:

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            # cv2.putText(frame, emotion, (right + 6, top - 6), font, 2.0, (0, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (left + 6, bottom +20), font, 1.0, (255, 255, 255), 1)
            identify1(frame, name, buf, buf_length, known_conf)
            emotion_count += 1
            if emotion_count >= 30:
                print("The Emotion Limit Reached More than 30 sec")
                threading.Thread(target=save_data_continuously, args=(name, detected_emotion)).start()
                print('The data will Store the Countiuesly for specific User')
                emotions_list = []
                
                for _ in range(30):
                    emotions_list.append(result[0]['dominant_emotion'])
                    # print(emotions_list)
                    # To display the empty list

                # Determine the dominant emotion based on counts
                dominant_emotion = Counter(emotions_list).most_common(1)[0][0]
                
                # print("This is my heart_rate ", heart_rate)
                print("This is my dominant Emotion", dominant_emotion)
                # now = datetime.now()
                # day = now.strftime('%d-%m-%Y')
                # attenance = 'Registered'
                # file_path = f"{day}.csv"
                # user_id = get_user_id_by_name(name)  
                # if user_id is not None:
                #     tryingheart(user_id)
                
                face_names.append(name)
                print(face_names)
                # with open(file_path, 'a', newline='') as f:
                #     write = csv.writer(f)
                #     for name in face_names:
                #             s1 += 1
                #             if s1 >= 5:
                #                 time = now.strftime("%H:%M:%S")
                #                 write.writerow([time,name,emotion,heartrate])
                #                 s1 = 0
                
                
             
                

            
            

        buf[i] = face_names
        i = (i + 1) % buf_length


        # print(buf)


        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

def dominant_emotion(request):
    data = Employee.objects.all()
    return render(request, 'app/dominant_emotion.html')


def landingpage(request):
    return render(request, 'app/landingpage.html')

def index(request):
    return render(request, 'app/test3.html')


def dashboard(request):
    data = Employee.objects.all()
    return render(request, 'app/dashboard.html', {'models': data})


def video_stream(request):
    stream()
    return HttpResponseRedirect(reverse('index'))


def add_photos(request):
	emp_list = Employee.objects.all()
	return render(request, 'app/add_photos.html', {'emp_list': emp_list})


def click_photos(request, emp_id):
	cam = cv2.VideoCapture(0)
	emp = get_object_or_404(Employee, id=emp_id)
	click(emp.name, emp.id, cam)
	return HttpResponseRedirect(reverse('add_photos'))


def train_model(request):
	trainer()
	return HttpResponseRedirect(reverse('index'))


def detected(request):
	if request.method == 'GET':
		date_formatted = datetime.today().date()
		date = request.GET.get('search_box', None)
		if date is not None:
			date_formatted = datetime.strptime(date, "%Y-%m-%d").date()
		det_list = Detected.objects.filter(time_stamp__date=date_formatted).order_by('time_stamp').reverse()

	# det_list = Detected.objects.all().order_by('time_stamp').reverse()
	return render(request, 'app/detected.html', {'det_list': det_list, 'date': date_formatted})


def identify(request):
	video_capture = cv2.VideoCapture(0)
	identify_faces(video_capture)
	return HttpResponseRedirect(reverse('index'))


def add_emp(request):
    if request.method == "POST":
        form = EmployeeForm(request.POST)
        if form.is_valid():
            emp = form.save()
            # post.author = request.user
            # post.published_date = timezone.now()
            # post.save()
            return HttpResponseRedirect(reverse('index'))
    else:
        form = EmployeeForm()
    return render(request, 'app/add_emp.html', {'form': form})


# def get_dominant_emotion(request, emotions_array):
#     emotion_counts = Counter(emotions_array)
#     dominant_emotion = max(emotion_counts, key=emotion_counts.get)
#     print('This is the', dominant_emotion)
#     return render(request, 'dominant_emotion')

#Heart-rate sensor code
from datetime import  datetime
import csv

# def tryingheart(user_id):
#     ser = serial.Serial("COM", 9600)
#     averageHeartRate = ser.readline().decode().strip()
#     print("This is the heartrate", averageHeartRate)
#     employee = Employee.objects.get(id=user_id)
#     ids = 0
#     now = datetime.now()
#     day = now.strftime('%d-%m-%Y')
#     attenance = 'Registered'
#     file_path = f"{day}.csv"
    
    
#     if employee:
#         name = employee.name
#         age = employee.age
#         email = employee.email
#         emotion = employee.emotion
#         heartrate = employee.heart_rate
#         average = heartrate
#         employee.heart_rate = averageHeartRate
#         heart_rate = float(employee.heart_rate)
#         attendance_List = []
#         file_path = f"{day}.csv"
#         with open(file_path, 'a', newline='') as f:
#             writer = csv.writer(f)
#             current_time = datetime.now().strftime("%H:%M:%S")
#             writer.writerow([current_time, name, emotion, heart_rate])
#             employee.save()
#         if age < 65 and heart_rate < 140:
#             subject = "Urgent Action Required: Individuals with Age < 65 and Heart Rate < 60"
#             message = f"""Dear {employee.name},

#         I hope this email finds you well.

#         I am writing to bring to your attention a matter of urgency regarding our MindWell project analysis. Our findings indicate that individuals under the age of 65 with a recorded heart rate of less than 60 beats per minute require immediate attention.

#         This demographic has been flagged due to potential health concerns associated with bradycardia. It is imperative that prompt action is taken to ensure the well-being of these individuals.

#         We recommend the following steps:

#         1. Conduct a thorough medical assessment to determine the underlying cause of the low heart rate.
#         2. Provide appropriate medical intervention and monitoring as necessary.
#         3. Offer support and guidance to the affected individuals and their caregivers.

#         Your cooperation in addressing this issue is crucial in preventing any adverse health outcomes. Please do not hesitate to reach out if you require further assistance or clarification on the recommended steps.

#         Thank you for your attention to this urgent matter.

#         Best regards,
#         [Shreyant Das]
#         [Founder & Ceo]"""
#             send_mail(
#                 subject,
#                 message,
#                 'mailjavasend@gmail.com',
#                 [email],
#                 fail_silently=False,
#             )
#             print("Send mail successfully in this based on this condition")

#         elif age >= 65 and heart_rate > 150:
#             subject = "Urgent Action Required: Individuals with Age ≥ 65 and Heart Rate > 150"
#             message = f"""Dear {employee.name},

#         I hope this email finds you well.

#         Our analysis from the MindWell project has revealed concerning findings regarding individuals aged 65 and above with a recorded heart rate exceeding 150 beats per minute.

#         This demographic requires urgent attention and intervention to prevent potential health complications. Elevated heart rates in this age group could indicate serious underlying cardiac issues that need immediate medical assessment and intervention.

#         We recommend the following actions:

#         1. Immediately conduct a thorough medical evaluation to assess the individual's cardiovascular health.
#         2. Provide urgent medical intervention and monitoring as necessary to stabilize the heart rate and address any underlying conditions.
#         3. Ensure continuous monitoring and follow-up care to prevent recurrence or complications.

#         Your prompt action in addressing this issue is critical in ensuring the well-being and safety of the affected individuals. Please reach out if you need any further assistance or guidance.

#         Thank you for your attention to this urgent matter. 

#         Best regards,
#         [Shreyant Das]
#         [Founder & Ceo]"""
#             send_mail(
#                 subject,
#                 message,
#                 'mailjavasend@gmail.com',
#                 [email],
#                 fail_silently=False,
#             )

#         elif emotion == 'sad' and heart_rate < 100:
#             subject = "Attention Required: Individuals Exhibiting Sadness and Heart Rate < 100"
#             message = f"""Dear {employee.name},

#         I hope this message finds you well.

#         It has come to our attention through the MindWell project analysis that certain individuals are exhibiting signs of sadness alongside a recorded heart rate of less than 100 beats per minute.

#         This combination of emotional distress and a relatively low heart rate requires careful consideration and support to address both mental and physical well-being.

#         We recommend the following steps:

#         1. Reach out to the affected individuals to offer emotional support and assistance.
#         2. Provide access to counseling services or mental health professionals for further evaluation and support.
#         3. Conduct a thorough medical assessment to investigate any underlying physiological factors contributing to the observed emotional state and heart rate.

#         Your prompt action in providing support and assistance to these individuals is crucial in ensuring their overall well-being. Please do not hesitate to reach out if you require any further assistance or guidance.

#         Thank you for your attention to this matter.
#         Best regards,
#         [Shreyant Das]
#         [Founder & Ceo]"""
#             send_mail(
#                 'Low Heart Rate Alert',
#                 f'Heart rate is low for employee {employee.name} and they are sad. Heart rate: {heart_rate}',
#                 'mailjavasend@gmail.com',
#                 [email],
#                 fail_silently=False,
#             )
#     ser.close()
#     return averageHeartRate


# def check_heart_rate(request):
    
#     employee = Employee.objects.first()  
#     if employee:
#         age = employee.age
#         emotion = employee.emotion
#         email = employee.email
#         heart_rate = int(employee.heart_rate)

       
#         if age < 65 and heart_rate > 140:
#             send_mail(
#                 'High Heart Rate Alert',
#                 f'Heart rate exceeded threshold for employee {employee.name}. Heart rate: {heart_rate}',
#                 'from@example.com',
#                 [email],
#                 fail_silently=False,
#             )

#         elif age >= 65 and heart_rate > 150:
#             send_mail(
#                 'High Heart Rate Alert',
#                 f'Heart rate exceeded threshold for employee {employee.name}. Heart rate: {heart_rate}',
#                 'from@example.com',
#                 [email],
#                 fail_silently=False,
#             )

       
#         elif emotion == 'Sad' and heart_rate < 100:
#             send_mail(
#                 'Low Heart Rate Alert',
#                 f'Heart rate is low for employee {employee.name} and they are sad. Heart rate: {heart_rate}',
#                 'from@example.com',
#                 [email],
#                 fail_silently=False,
#             )

#     return render(request, "app/dashboard.html")
