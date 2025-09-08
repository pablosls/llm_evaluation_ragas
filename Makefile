.PHONY: install rag bench ragas test

install:
	pip install -e ".[dev]"

rag:
	pip install -e ".[rag]"

bench:
	python examples/bench_speed.py

ragas:
	python -m pip install ragas datasets
	python examples/eval_ragas.py

test:
	pytest

compare:
	python examples/compare_models.py

compare:
	python -m pip install ragas datasets
	python examples/compare_models.py

compare:
	python -m pip install ragas datasets
	python examples/compare_models.py
