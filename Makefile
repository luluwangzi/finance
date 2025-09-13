.PHONY: dev install run docker-build docker-run docker-stop hf-deploy

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

hf-deploy:
	@if [ -z "$$HF_TOKEN" ] || [ -z "$$HF_USERNAME" ]; then \
		echo "Please export HF_TOKEN and HF_USERNAME before deploying."; \
		echo "Create token at https://huggingface.co/settings/tokens"; \
		exit 2; \
	fi
	python3 -m pip install --break-system-packages huggingface_hub >/dev/null 2>&1 || true
	python3 scripts/deploy_hf_space.py

