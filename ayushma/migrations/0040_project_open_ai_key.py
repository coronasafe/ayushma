# Generated by Django 4.2.1 on 2023-08-31 21:16

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ayushma", "0039_alter_document_uploading"),
    ]

    operations = [
        migrations.AddField(
            model_name="project",
            name="open_ai_key",
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
