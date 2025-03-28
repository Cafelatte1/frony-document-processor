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
|    | Huang&Harper(2009)[14]         | semi-supervised        | 91.3      |
|    | McCloskyetal. (2006)[26]       | semi-supervised        | 92.1      |
|    | Vinyals&Kaiserelal. (2014)[37] | semi-supervised        | 92.1      |
|  3 | Transformer(4layers)           | semi-supervised        | 92.7      |
|  4 | Luongetal. (2015)[23]          | multi-task             | 93.0      |
|    | Dyeretal. (2016)[8]            | generative             | 93.3      |

Vinyals & Kaiser el al. (2014) [37] WSJ only, discriminative 88.3
```

### PDF Page Search for LLM based chunking
* LLM based chunking is advanced technique for 

## Easy & Convenient Usage
```
from frony_document_processor.parser import ParserTXT
from frony_document_processor.parser import ParserPDF
from frony_document_processor.parser import ParserPPTX
from frony_document_processor.parser import ParserPDFImage
from frony_document_processor.parser import ParserImage

from frony_document_processor.chunker import RuleBasedTextChunker
from frony_document_processor.chunker import LLMBasedTextChunker
from frony_document_processor.chunker import LLMBasedImageChunker

from frony_document_processor.embedder import OpenAIEmbedder
from frony_document_processor.embedder import SentenceTransformerEmbedder
```


## Auto Table Extraction for PDF



## sfsdf

[Content]

LangChain helps developers build applications powered by LLMs through a standard
interface for models, embeddings, vector stores, and more. 

Use LangChain for:
- **Real-time data augmentation**. Easily connect LLMs to diverse data sources and
external / internal systems, drawing from LangChain’s vast library of integrations with
model providers, tools, vector stores, retrievers, and more.
- **Model interoperability**. Swap models in and out as your engineering team
experiments to find the best choice for your application’s needs. As the industry
frontier evolves, adapt quickly — LangChain’s abstractions keep you moving without
losing momentum.

## Installation & Prequirements
While the LangChain framework can be used standalone, it also integrates seamlessly
with any LangChain product, giving developers a full suite of tools when building LLM
applications. 

To improve your LLM application development, pair LangChain with:

- [LangSmith](http://www.langchain.com/langsmith) - Helpful for agent evals and
observability. Debug poor-performing LLM app runs, evaluate agent trajectories, gain
visibility in production, and improve performance over time.
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Build agents that can
reliably handle complex tasks with LangGraph, our low-level agent orchestration
framework. LangGraph offers customizable architecture, long-term memory, and
human-in-the-loop workflows — and is trusted in production by companies like LinkedIn,
Uber, Klarna, and GitLab.
- [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#langgraph-platform) - Deploy
and scale agents effortlessly with a purpose-built deployment platform for long
running, stateful workflows. Discover, reuse, configure, and share agents across
teams — and iterate quickly with visual prototyping in
[LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/).

## Additional resources
- [Tutorials](https://python.langchain.com/docs/tutorials/): Simple walkthroughs with
guided examples on getting started with LangChain.
- [How-to Guides](https://python.langchain.com/docs/how_to/): Quick, actionable code
snippets for topics such as tool calling, RAG use cases, and more.
- [Conceptual Guides](https://python.langchain.com/docs/concepts/): Explanations of key
concepts behind the LangChain framework.
- [API Reference](https://python.langchain.com/api_reference/): Detailed reference on
navigating base packages and integrations for LangChain.
