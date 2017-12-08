from django.db import models

# Create your models here.

class Info(models.Model):

    ID = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=100)
    description = models.TextField()
    imgPath = models.FilePathField()

    def __unicode__(self):
        return self.ID