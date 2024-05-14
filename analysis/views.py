from django.shortcuts import render

# Create your views here.
# views.py
from django.db.models import Avg, Count
from django.core.mail import send_mail
from collections import Counter

from app.models import Employee

# 1. Select Emotions for a Particular `reg_id`
def get_emotions_for_reg_id(reg_id):
    emotions_for_reg_id = Employee.objects.filter(reg_id=reg_id).values_list('dominant_emotion', flat=True)
    return list(emotions_for_reg_id)

# 2. Store Emotions in an Array
def store_emotions_in_array(emotions_array, reg_id):
    data = Employee.objects.get(reg_id=reg_id)
    data.emotions_array = emotions_array
    data.save()

# 3. Check for Dominant Emotion
def get_dominant_emotion(emotions_array):
    emotion_counts = Counter(emotions_array)
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    return dominant_emotion

# 4. Store Dominant Emotion in the Database
def store_dominant_emotion(reg_id, dominant_emotion):
    data = Employee.objects.get(reg_id=reg_id)
    data.final_emotion = dominant_emotion
    data.save()

# 5. Trigger Email based on Final Emotion
def trigger_email_on_emotion(dominant_emotion):
    if dominant_emotion in ['sad', 'angry', 'fear', 'disgust']:
        send_email()

# 6. Select Average Heart Rate for a Particular `reg_id`
def get_avg_heart_rate(reg_id):
    avg_heart_rate = HeartRate.objects.filter(reg_id=reg_id).aggregate(avg_heart_rate=Avg('heartbeat'))['avg_heart_rate']
    return avg_heart_rate

# 7. Check Against Threshold and Trigger Email
def check_and_trigger_email(data):
    threshold = 140 if data.age <= 65 else 150
    if data.final_heart_rate > threshold:
        trigger_email_on_heart_rate()

# Email Sending Function
def trigger_email_on_heart_rate():
    subject = 'High Heart Rate Alert'
    message = 'This is an alert email due to high heart rate detected.'
    from_email = 'your_email@example.com'
    recipient_list = ['recipient@example.com']
    
    send_mail(subject, message, from_email, recipient_list)
