from django.urls import path

from . import views

urlpatterns = [
    path('index/', views.index, name='index'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('add_photos/', views.add_photos, name='add_photos'),
    path('add_photos/<slug:emp_id>/', views.click_photos, name='click_photos'),
    path('train_model/', views.train_model, name='train_model'),
    path('detected/', views.detected, name='detected'),
    path('identify/', views.identify, name='identify'),
    path('add_emp/', views.add_emp, name='add_emp'),
    path('show_database/', views.show_database, name='show_database'),
 
    path('dashboard/', views.dashboard, name='dashboard'),
    path('', views.landingpage, name='landingpage'),
    # path('tryingheart/',views.tryingheart,name='tryingheart'),
    path('daily_records/', views.daily_records, name='daily_records'),   
]