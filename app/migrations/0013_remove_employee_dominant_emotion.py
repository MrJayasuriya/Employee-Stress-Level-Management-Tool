# Generated by Django 4.2.8 on 2024-03-01 07:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0012_employee_dominant_emotion'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='employee',
            name='dominant_emotion',
        ),
    ]
