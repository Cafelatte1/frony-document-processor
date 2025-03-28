# Frony Doument Processor
[Content]

## Why use Frony Doument Processor?
### Image Parsing for PPTX, PDF
> [!NOTE]
> Libreoffice should be installed for ParserPPTX
* Parse PPTX and PDF files as images and output base64-encoded data for LLMs.
```python
parser = ParserPPTX()
df = parser.parse("test_files/test_pptx.pptx")
df
```
```
page_number	page_content
1	iVBORw0KGgoAAAANSUhEUgAAD6EAAAjKCAIAAADiFw3ZAA...
2	iVBORw0KGgoAAAANSUhEUgAAD6EAAAjKCAIAAADiFw3ZAA...
3	iVBORw0KGgoAAAANSUhEUgAAD6EAAAjKCAIAAADiFw3ZAA...
```

### Auto Table Extraction for PDF
* The in-built algorithm extracts tables in markdown style, which works well for LLMs.
```python
# Attention is all you need
from frony_document_processor.parser import ParserPDF
parser = ParserPDF()
df = parser.parse("test_files/test_pdf.pdf")
df["page_content"].iloc[-6]
```
```
Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23
of WSJ)
Parser Training WSJ 23 F1

|    | Parser                         | Training               | WSJ23F1   |
|---:|:-------------------------------|:-----------------------|:----------|
|  0 | Vinyals&Kaiserelal. (2014)[37] | WSJonly,discriminative | 88.3      |
|    | Petrovetal. (2006)[29]         | WSJonly,discriminative | 90.4      |
|    | Zhuetal. (2013)[40]            | WSJonly,discriminative | 90.4      |
|    | Dyeretal. (2016)[8]            | WSJonly,discriminative | 91.7      |
|  1 | Transformer(4layers)           | WSJonly,discriminative | 91.3      |
|  2 | Zhuetal. (2013)[40]            | semi-supervised        | 91.3      |
```

### PDF Page Search for LLM based chunking
* LLM-based chunking is an advanced technique for RAG
* When using this approach, there is a key challenge is determining where a chunk originates
* The **jaccard similarity score** and **relative positional score** are used for scoring
```python
parser = ParserPDF()
df = parser.parse("test_files/test_pdf.pdf")

# n_gram is the number of words composing a keyword 
chunker = LLMBasedTextChunker(n_gram=2)
chunks = chunker.chunk(
    df,
    splitter_config=[
        {"type": "llm_text", "params": {"chunk_size": 2048, "chunk_overlap": 2048 // 4}},
    ]
)
# First chunk is always total length of chunks
total_chunks = next(chunks)

df_chunk = []
for chunk in chunks:
    df_chunk.append(chunk)
df_chunk = pd.DataFrame(df_chunk)
df_chunk
```

