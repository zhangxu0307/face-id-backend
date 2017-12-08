#encoding=utf-8
from django.conf.urls import url
from . import views

urlpatterns = [
    # 识别
    url(r'^recognition/$', views.FaceRecognition.as_view(), name='recognition'),
    # 注册
    url(r'^register/$', views.Register.as_view(), name='register'),
    # 删除记录
    url(r'^delete/$', views.DeleteFace.as_view(), name='delete'),

]