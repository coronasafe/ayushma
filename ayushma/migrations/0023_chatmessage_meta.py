# Generated by Django 4.2.1 on 2023-05-29 07:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ayushma", "0022_merge_20230526_1025"),
    ]

    operations = [
        migrations.AddField(
            model_name="chatmessage",
            name="meta",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
