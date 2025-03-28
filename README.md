# Frony Doument Processor for RAG
The Frony Document Processor for RAG is designed to facilitate seamless document processing for large language models (LLMs) by transforming various document formats into structured data and images. It provides a suite of tools to parse, extract, and manage content from PPTX and PDF files, enabling the efficient handling of documents for downstream applications such as Retrieval-Augmented Generation (RAG).
```bash
pip install git+https://github.com/Cafelatte1/frony-document-processor
```

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

### PDF Page Search Algorithm for LLM based chunking
* LLM-based chunking is an advanced technique for RAG
* When using this approach, there is a key challenge is determining where a chunk originates
* The **jaccard similarity score** and **relative positional score** are used for scoring
```python
parser = ParserPDF()
df = parser.parse("test_files/test_pdf.pdf")

# n_gram is the number of words composing a keyword for calculating jaccard similarity score
chunker = LLMBasedTextChunker(n_gram=2)
chunks = chunker.chunk(
    df,
    # 'params' key is the parameters for langchain RecursiveCharacterTextSplitter
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
# Chunks is generated after concatenating page contents
# Algorithm searches pdf page for each chunks
df_chunk
```
```
page_number	chunk_type	chunk_id	chunk_content
1	llm_text	0	### 1. 연구 배경 및 목적\n- 기존의 시퀀스 변환 모델들은 복잡한 순환 신경...
1	llm_text	0	### 모델 성능 및 효율성\n- 제안된 모델은 WMT 2014 영어-독일어 번역 ...
1	llm_text	0	### 1. 서론\n- **재귀 신경망**: 재귀 신경망(RNN)과 장단기 기억(L...
2	llm_text	0	### 1. Transformer 모델 소개\n- **주요 내용**: Transfo...
2	llm_text	0	### 1. 텍스트 함의 및 문장 표현\n- 텍스트 함의와 학습 과제에 독립적인 문...
3	llm_text	0	### 1. Transformer 모델 아키텍처\n- Transformer는 인코더...
3	llm_text	0	### 1. 스케일된 점곱 주의 (Scaled Dot-Product Attentio
```
## Best Practice
### Vector DB with yielded chunks
* Invert the vector to DB in batches, not all at once
* Users can search documents before the entire dataset is fully processed
```python
async def fn_process(page_conatiner, chunkers, doc_id, db, collection, batch_size=4):
    max_progress_value = 100.0 / len(chunkers)
    for idx, chunker in enumerate(chunkers):
        # Frony Document Manager - Chunker
        chunk_generator = chunker.chunk(page_conatiner)
        total_chunks = next(chunk_generator)
        data = []
        for chunk in chunk_generator:
            data.append(chunk)
            if len(data) >= batch_size:
                # Invert the vector to DB in batches, not all at once
                if await fn_insert_vector(data, doc_id, collection):
                    await fn_update_progress(data, doc_id, total_chunks, max_progress_value, db)
                data = []
        if data:
            if await fn_insert_vector(data, doc_id, collection):
                await fn_update_progress(None, doc_id, total_chunks, max_progress_value * (idx + 1), db)
```
