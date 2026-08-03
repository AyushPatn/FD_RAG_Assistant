import sys
import os

# Ensure src is on path for tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))


def test_rag_pipeline_exists():
    base = os.path.join(os.path.dirname(__file__), os.pardir, 'src')
    path = os.path.join(base, 'rag_pipeline.py')
    assert os.path.exists(path), f"Expected {path} to exist"


def test_ragas_evaluation_helper_exists():
    import ragas_integration

    assert hasattr(ragas_integration, 'build_evaluation_dataset')
    assert hasattr(ragas_integration, 'run_ragas_evaluation')
