"""Smoke test verifying baseline environment and dependency imports."""

def test_imports():
    import streamlit
    import pypdf
    import sklearn
    import networkx
    import numpy

    assert streamlit.__version__
    assert pypdf.__version__
    assert sklearn.__version__
    assert networkx.__version__
    assert numpy.__version__
