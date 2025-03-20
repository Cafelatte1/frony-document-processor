SEED = 42
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from openai import OpenAI
import Levenshtein
from dotenv import load_dotenv
load_dotenv()

class RuleBasedTextChunker():
    def __init__(
            self, tokenizer_path=None,
    ):
        if tokenizer_path is None:
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def compress_text(x, max_text_len=2000):
        return "".join(re.sub(r"[^a-z가-힣0-9]", "", x.lower()).split())[:max_text_len]

    def search_page_number(self, query, page_container, n_buckets=10, n_characters=50):
        # algorithm: text matching
        # complexity: O(n)
        reference = page_container["page_content"].apply(self.compress_text).to_list()
        compressed_query = self.compress_text(query)
        splitted_query = ["".join(i)[:n_characters] for i in np.array_split(list(compressed_query), n_buckets)]
        score = pd.Series([sum([q in r for q in splitted_query]) for r in reference], index=page_container["page_number"].to_list())
        return score.sort_values(ascending=False)
    
    def chunk(
            self, page_container, page_separator="\n\n",
            splitter_config=[
                {"type": "rule_short", "params": {"chunk_size": 384, "chunk_overlap": 384 // 4, "separators": [""]}},
                {"type": "rule_long", "params": {"chunk_size": 768, "chunk_overlap": 768 // 4, "separators": [""]}},
            ],
    ):
        if self.tokenizer is None:
            splitter_container = {
                config["type"]: RecursiveCharacterTextSplitter(**config["params"])
                for config in splitter_config
            }
        else:
            splitter_container = {
                config["type"]: RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, **config["params"])
                for config in splitter_config
            }

        texts = page_separator.join(page_container["page_content"])
        chunk_container = {}
        total_chunks = 0
        for chunk_type, splitter in splitter_container.items():
            chunk_container[chunk_type] = splitter.split_text(texts)
            total_chunks += len(chunk_container[chunk_type])
        yield total_chunks
        
        for chunk_type, chunks in chunk_container.items():
            for chunk_id, chunk_data in enumerate(tqdm(chunks, desc=f"create documents... ({chunk_type})")):
                chunk_data = chunk_data.strip()
                # searching page number
                score = self.search_page_number(chunk_data, page_container)
                # create output
                output = {
                    "page_number": score.index[0],
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                    "chunk_content": chunk_data.strip(),
                }
                yield output

    
class LLMBasedTextChunker():
    def __init__(
            self, tokenizer_path=None,
            llm_server_config={"base_url": "http://localhost:9001/v1", "api_key": "token-abc123"},
            llm_model_type="openai", llm_model_name="default", llm_n_trials=5, llm_max_len=8192, llm_max_tokens=1024, llm_min_tokens=16,
    ):
        if tokenizer_path is None:
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if llm_model_type == "openai":
            self.client = OpenAI()
            self.sampling_params = {
                "max_completion_tokens": llm_max_tokens,
                "n": 1,
                "temperature": 0.5,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
            }
        else:
            self.client = OpenAI(**llm_server_config)
            self.sampling_params = {
                "max_tokens": llm_max_tokens,
                "min_tokens": llm_min_tokens,
                "truncate_prompt_tokens": llm_max_len - llm_max_tokens,
                "n": 2,
                "best_of": 3,
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
            }
        self.llm_model_name = llm_model_name
        self.llm_max_trials = llm_n_trials
        self.prompt_template = {
            "system": """
        """,
            "user": """
다음의 글을 주제별로 나누어 자세하게 요약해 주세요.

{context}
        """
        }


    @staticmethod
    def compress_text(x, max_text_len=2000):
        return "".join(re.sub(r"[^a-z가-힣0-9]", "", x.lower()).split())[:max_text_len]

    def search_page_number(self, query, page_container):
        # algorithm: levenshtein
        # complexity: O(n)
        reference = page_container["page_content"].apply(self.compress_text).to_list()
        compressed_query = self.compress_text(query)
        score = pd.Series([Levenshtein.ratio(compressed_query, r) for r in reference], index=page_container["page_number"].to_list()).sort_values(ascending=False)
        return score
    
    def chunk(
            self, page_container, page_separator="\n\n",
            splitter_config=[
                {"type": "llm_base", "params": {"chunk_size": 3072, "chunk_overlap": 3072 // 4, "separators": [""]}},
            ],
    ):
        if self.tokenizer is None:
            splitter_container = {
                config["type"]: RecursiveCharacterTextSplitter(**config["params"])
                for config in splitter_config
            }
        else:
            splitter_container = {
                config["type"]: RecursiveCharacterTextSplitter.from_huggingface_tokenizer(self.tokenizer, **config["params"])
                for config in splitter_config
            }

        texts = page_separator.join(page_container["page_content"])
        chunk_container = {}
        total_chunks = 0
        for chunk_type, splitter in splitter_container.items():
            chunk_container[chunk_type] = splitter.split_text(texts)
            total_chunks += len(chunk_container[chunk_type])
        total_chunks *= self.sampling_params["n"]
        yield total_chunks

        for chunk_type, chunks in chunk_container.items():
            for chunk_id, chunk_data in enumerate(tqdm(chunks, desc=f"create documents... ({chunk_type})")):
                # generation
                cnt = 0
                while cnt < self.llm_max_trials:
                    try:
                        completion = self.client.chat.completions.create(
                            model=self.llm_model_name,
                            messages=[
                                {"role": "user", "content": self.prompt_template["user"].format(context=chunk_data.strip()).strip()},
                            ],
                            extra_body=self.sampling_params,
                            seed=SEED,
                        )
                        break
                    except Exception as e:
                        completion = None
                        cnt += 1
                        print(f"ERROR in generation -> retry / msg={e}, iteration={cnt}")
                        continue
                if completion is None:
                    print(f"nothing generated -> skip chunking")
                    continue
                for chunk_id, cmpl in enumerate(completion.choices):
                    gened_data = cmpl.message.content
                    # searching page number
                    score = self.search_page_number(gened_data, page_container)
                    output = {
                        "page_number": score.index[0],
                        "chunk_type": "image",
                        "chunk_id": chunk_id,
                        "chunk_content": gened_data.strip(),       
                    }
                    yield output


class LLMBasedImageChunker():
    def __init__(
            self,
            llm_server_config={"base_url": "http://localhost:9001/v1", "api_key": "token-abc123"},
            llm_model_type="openai", llm_model_name="default", llm_n_trials=5, llm_max_len=8192, llm_max_tokens=1024, llm_min_tokens=16,
            
        ):
        if llm_model_type == "openai":
            self.client = OpenAI()
            self.sampling_params = {
                "max_completion_tokens": llm_max_tokens,
                "n": 1,
                "temperature": 0.5,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
            }
        else:
            self.client = OpenAI(**llm_server_config)
            self.sampling_params = {
                "max_tokens": llm_max_tokens,
                "min_tokens": llm_min_tokens,
                "truncate_prompt_tokens": llm_max_len - llm_max_tokens,
                "n": 2,
                "best_of": 3,
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.95,
                "frequency_penalty": 0.5,
            }
        self.llm_model_name = llm_model_name
        self.llm_max_trials = llm_n_trials
        self.prompt_template = {
            "system": """
        """,
            "user": """
이미지의 내용을 주제별로 나누어 자세하게 요약해 주세요.
        """
        }
    
    def chunk(self, page_container):
        total_chunks = len(page_container)
        total_chunks *= self.sampling_params["n"]
        yield total_chunks

        for idx, row in page_container.iterrows():
            # generation
            cnt = 0
            while cnt < self.llm_max_trials:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": self.prompt_template["user"].strip()},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{row['page_content']}"}}
                                ]
                            }
                        ],
                        extra_body=self.sampling_params,
                    )
                    break
                except Exception as e:
                    completion = None
                    cnt += 1
                    print(f"ERROR in generation -> retry / msg={e}, iteration={cnt}")
                    continue
            if completion is None:
                print(f"nothing generated -> skip chunking")
                continue
            for chunk_id, cmpl in enumerate(completion.choices):
                gened_data = cmpl.message.content
                output = {
                    "page_number": row["page_number"],
                    "chunk_type": "image",
                    "chunk_id": chunk_id,
                    "chunk_content": gened_data.strip(),  
                }
                yield output
