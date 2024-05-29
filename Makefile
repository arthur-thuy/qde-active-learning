format:
	isort src/ --profile black
	black src/

lint:
	# pydocstyle src/
	flake8 src/ 