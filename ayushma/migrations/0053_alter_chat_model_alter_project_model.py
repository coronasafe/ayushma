# Generated by Django 4.2.6 on 2024-07-02 16:07

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("ayushma", "0052_document_failed"),
    ]

    operations = [
        migrations.AlterField(
            model_name="chat",
            name="model",
            field=models.IntegerField(
                blank=True,
                choices=[
                    (1, "Gpt 3 5"),
                    (2, "Gpt 3 5 16K"),
                    (3, "Gpt 4"),
                    (4, "Gpt 4 32K"),
                    (5, "Gpt 4 Visual"),
                    (6, "Gpt 4 Turbo"),
                    (7, "Gpt 4 Omni"),
                ],
                null=True,
            ),
        ),
        migrations.AlterField(
            model_name="project",
            name="model",
            field=models.IntegerField(
                choices=[
                    (1, "Gpt 3 5"),
                    (2, "Gpt 3 5 16K"),
                    (3, "Gpt 4"),
                    (4, "Gpt 4 32K"),
                    (5, "Gpt 4 Visual"),
                    (6, "Gpt 4 Turbo"),
                    (7, "Gpt 4 Omni"),
                ],
                default=1,
            ),
        ),
    ]