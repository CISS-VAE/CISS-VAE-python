# Configuration file for the Sphinx documentation builder.

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
print(">> DEBUG: Inserting src into sys.path:", os.path.abspath("../../src"))
sys.path.insert(0, os.path.abspath("../../src"))  # so autodoc can find your package

#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'CISS-VAE'
copyright = '2025, Yasin Khadem Charvadeh, Danielle Vaithilingam'
author = 'Yasin Khadem Charvadeh, Danielle Vaithilingam'
release = '1.0.3'


# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",         # for Google/NumPy docstrings
    "sphinx.ext.viewcode",         # link to source
    "sphinx.ext.autosummary",      # create API summary
    "sphinx.ext.intersphinx",      # cross-project refs
    "sphinx_autodoc_typehints",    # better type hint rendering
    "myst_parser",                 # if using markdown
    "sphinx_copybutton"
]

autosummary_generate = True
add_module_names = False  # so :func:`train_model` instead of :func:`vae.train_model`




templates_path = ['_templates']
exclude_patterns = []

# copybutton behavior (adjust to preference)
copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False
copybutton_remove_prompts = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_css_files = [
    'styles.css',
]


autodoc_mock_imports = ["torch", "sklearn"]  # add libs you don't need at doc-build time
