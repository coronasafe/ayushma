# Generated by Django 4.2.1 on 2023-08-10 07:12

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ayushma", "0036_document_uploading"),
    ]

    operations = [
        migrations.AddField(
            model_name="chat",
            name="model",
            field=models.IntegerField(
                blank=True,
                choices=[
                    (1, "Gpt 3 5"),
                    (2, "Gpt 3 5 16K"),
                    (3, "Gpt 4"),
                    (4, "Gpt 4 32K"),
                ],
                null=True,
            ),
        ),
    ]
