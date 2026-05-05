.PHONY: install test lint clean

install:
# 	pip install -e .
# 	pip install -r requirements-dev.txt  # для разработки 

test:
	python -m unittest tests.test_cg_framework

lint:
	flake8 src/ tests/
	black --check src/ tests/

clean:
# 	find . -type d -name "__pycache__" -exec rm -rf {} +
# 	rm -rf .pytest_cache

clean_records:
	rm -rf records/* logs/*

test_launch: clean_records
	python main.py --file heat_aurum.in -k 4 -i 10000 -m 500 --solver layer


test_launch_fcc: clean_records
	python launcher.py --file heat_aurum.in -i 3000 -m 500 --solver fcc


test_launch_fcc_small: clean_records
	python launcher.py --file heat_aurum_small.in -i 10000 -m 250 --solver fcc

test_crack: clean_records
	python crack_launcher.py --file tests/crack_ni_lg_velocity_set.in -i 10000 -m 250 -s 2 --solver fcc