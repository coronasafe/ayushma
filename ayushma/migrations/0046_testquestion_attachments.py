# Generated by Django 4.2.5 on 2023-10-15 10:23

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ayushma", "0045_testrun_references"),
    ]

    operations = [
        migrations.AddField(
            model_name="testquestion",
            name="attachments",
            field=models.ManyToManyField(blank=True, to="ayushma.document"),
        ),
    ]
