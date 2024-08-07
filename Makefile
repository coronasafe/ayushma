.PHONY: build, re-build, up, down, list, logs, test, makemigrations


DOCKER_VERSION := $(shell docker --version 2>/dev/null)

env ?= local
docker_config_file ?= docker-compose.$(env).yaml

all:
ifndef DOCKER_VERSION
    $(error "command docker is not available, please install Docker")
endif

re-build:
	docker compose -f $(docker_config_file) build --no-cache

build:
	docker compose -f $(docker_config_file) build

up:
	docker compose -f $(docker_config_file) up -d

down:
	docker compose -f $(docker_config_file) down

list:
	docker compose -f $(docker_config_file) ps

logs:
	docker compose -f $(docker_config_file) logs

makemigrations: up
	docker compose -f $(docker_config_file) exec django bash -c "python manage.py makemigrations"

migrate: up
	docker compose -f $(docker_config_file) exec django bash -c "python manage.py migrate"

checkmigration:
	docker compose -f $(docker_config_file) exec django bash -c "python manage.py makemigrations --check --dry-run"

test: up
	docker exec django bash -c "python manage.py test --keepdb --parallel=$(nproc)"

test_coverage: up
	docker exec django bash -c "coverage run manage.py test --settings=config.settings.test --keepdb --parallel=$(nproc)"
	docker exec django bash -c "coverage combine || true; coverage report"
