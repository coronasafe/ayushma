# Generated by Django 4.1.7 on 2023-05-05 18:32

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ayushma", "0013_resetpasswordtoken"),
    ]

    operations = [
        migrations.AddField(
            model_name="document",
            name="document_type",
            field=models.CharField(
                choices=[("file", "File"), ("url", "URL"), ("text", "Text")],
                default="file",
                max_length=4,
            ),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="document",
            name="text_content",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name="document",
            name="file",
            field=models.FileField(blank=True, null=True, upload_to="documents/"),
        ),
    ]
