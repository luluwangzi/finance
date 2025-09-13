.PHONY: dev install run docker-build docker-run docker-stop

dev: install run

install:
	python3 -m pip install --break-system-packages -r requirements.txt

run:
	bash start.sh

docker-build:
	docker build -t mdd-csp-app:latest .

docker-run:
	docker run --rm -p 8501:8501 --name mdd-csp-app mdd-csp-app:latest

docker-stop:
	-docker stop mdd-csp-app

