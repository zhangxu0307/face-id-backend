#encoding=utf-8
from django.conf.urls import url
from . import views

urlpatterns = [
    # 识别
    url(r'^recongition/$', views.FaceRecognition.as_view(), name='recongition'),
    # 注册
    url(r'^register/$', views.Register.as_view(), name='register'),

]