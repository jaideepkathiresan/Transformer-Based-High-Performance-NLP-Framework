.PHONY: clean build test demo

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "*.so" -delete
	find . -name "__pycache__" -delete
	find . -name "*.pyc" -delete

build:
	pip install -e .

test:
	python -m unittest discover tests/

demo:
	python examples/scale_demo.py

profile:
	python -m cProfile -o profile.stats examples/generate_demo.py
