# Generated by Django 4.2.8 on 2024-02-29 13:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0010_employee_age_employee_contact_number_employee_email_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='employee',
            name='heart_rate',
            field=models.CharField(default=2, max_length=200),
            preserve_default=False,
        ),
    ]
