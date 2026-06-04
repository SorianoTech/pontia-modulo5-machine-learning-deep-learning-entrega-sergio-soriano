from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

project = "Proyecto Final ML - Cancelacion de Reservas Hoteleras"
author = "Sergio Soriano"
copyright = "2026, Sergio Soriano"
language = "es"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "undoc-members": False,
    "exclude-members": "_get_metadata_request,get_metadata_routing,set_score_request",
}

autodoc_mock_imports = [
    "catboost",
    "httpx",
    "mlflow",
    "streamlit",
    "tensorflow",
    "xgboost",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_title = project
html_static_path = []
