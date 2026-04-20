import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath("../../src"))

project = 'CISS-VAE'
copyright = '2025, Yasin Khadem Charvadeh, Danielle Vaithilingam, Kenneth Seier, Katherine S. Panageas, Mithat Gönen, Yuan Chen'
author = 'Yasin Khadem Charvadeh, Danielle Vaithilingam, Kenneth Seier, Katherine S. Panageas, Mithat Gönen, Yuan Chen'
release = '1.0.18'

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton"
]

autosummary_generate = True
add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# ---- CRITICAL FIX ----
nb_execution_mode = "off"
nb_execution_timeout = 300
nb_execution_cache_path = ".jupyter_cache"
nb_execution_allow_errors = True

# Napoleon
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = []

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = ['styles.css']

autodoc_mock_imports = [
    "torch",
    "hdbscan",
    "optuna",
]

nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}