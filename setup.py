from setuptools import setup

setup(
    name="frony-document-processor",
    version="0.5.0",
    packages=["frony_document_processor"],
    install_requires=[
        "numpy>=2.2.4",
        "pandas>=2.2.3",
        "python-dotenv>=1.1.0",
        "transformers>=4.50.1",
        "langchain-text-splitters==0.3.7",
        "openai>=1.68.2",
        "pdfplumber>=0.11.5",
        "pillow>=11.1.0",
        "tabulate>=0.9.0",
        "sentence-transformers>=4.0.1",
        "tqdm>=4.67.1",
    ],
)
