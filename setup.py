from setuptools import setup, find_packages

setup(
    name="frony-document-processor",
    version="0.2.0",
    packages=["frony_document_processor"],
    install_requires=[
        "numpy",
        "pandas",
        "python-dotenv",
        "transformers",
        "langchain-text-splitters",
        "levenshtein",
        "openai",
        "pdfplumber",
        "Pillow",
        "tabulate"
    ],
)
