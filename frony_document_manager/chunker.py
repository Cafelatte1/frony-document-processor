import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class RuleBasedTextChunker():
    def __init__(
        self,
        tokenizer_path=None,
        search_page_number_params={"n_buckets": 10, "n_characters": 50, "max_text_len": 2000},
    ):
        if tokenizer_path is None:
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.search_page_number_params = search_page_number_params

    def compress_text(self, x):
        return "".join(re.sub(r"[^a-z가-힣0-9]", "", x.lower()).split())[:self.search_page_number_params["max_text_len"]]

    def search_page_number(self, query, page_container):
        # algorithm: text matching
        # complexity: O(n)
        reference = page_container["page_content"].apply(self.compress_text).to_list()
        compressed_query = self.compress_text(query)
        splitted_query = [
            "".join(i)[:self.search_page_number_params["n_characters"]]
            for i in np.array_split(list(compressed_query), self.search_page_number_params["n_buckets"])
        ]
        score = pd.Series([sum([q in r for q in splitted_query]) for r in reference], index=page_container["page_number"].to_list())
        return score.sort_values(ascending=False)
    
    def chunk(
            self, page_container, page_separator="\n\n",
            splitter_config=[
                {"type": "rule_short", "params": {"chunk_size": 128, "chunk_overlap": 128 // 4}},
                {"type": "rule_long", "params": {"chunk_size": 512, "chunk_overlap": 512 // 4}},
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
                    "page_number": int(score.index[0]),
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                    "chunk_content": chunk_data.strip(),
                }
                yield output

    
class LLMBasedTextChunker():
    def __init__(
        self,
        tokenizer_path=None,
        search_page_number_params={"max_text_len": 2000},
        llm_model_type="openai",
        llm_server_config={},
        llm_model_name="gpt-4o-mini",
        sampling_params={"max_completion_tokens": 1024, "n": 1, "temperature": 0.5, "top_p": 0.95},
        max_trials=5,
        n_gram=2,
        prompt_template={
            "system": '',
            "user": """
다음의 글을 주제별로 나누어 한국어로 요약해 주세요.

{context}
"""
        },
        seed=42,
    ):
        if tokenizer_path is None:
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.search_page_number_params = search_page_number_params
        self.llm_model_type = llm_model_type
        self.client = OpenAI(**llm_server_config)
        self.sampling_params = sampling_params
        self.llm_model_name = llm_model_name
        self.max_trials = max_trials
        self.n_gram = n_gram
        self.prompt_template = prompt_template
        self.seed = seed

    @staticmethod
    def create_ngrams(words, n, separator="|"):
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = separator.join(words[i:i + n])
            ngrams.append(ngram)
        return ngrams

    @staticmethod
    def calc_jaccard_score(data1, data2):
        set1 = set(data1)
        set2 = set(data2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union != 0 else 0

    @staticmethod
    def create_linear_decay_vector(vector_length, center_point, min_val=0.1, max_val=1):
        if center_point < 1:
            return np.array([])
        left_half = np.linspace(min_val, max_val, center_point, endpoint=True)
        if len(left_half) == 1:
            left_half = np.array([max_val])
        right_half = np.linspace(max_val, min_val, vector_length - center_point + 1, endpoint=False)[1:]
        if len(right_half) == 1:
            right_half = np.array([min_val])
        return np.concatenate([left_half, right_half])

    def search_page_number(self, query, page_container, linear_decay_vector):
        token_reference = page_container["page_content"].apply(
            lambda x: self.create_ngrams(self.tokenizer.tokenize(x), n=self.n_gram) if self.tokenizer is not None
            else self.create_ngrams(" ".join(x.split()).split(), n=self.n_gram)
        ).to_list()
        if self.tokenizer is None:
            token_query = self.create_ngrams(" ".join(query.split()).split(), n=self.n_gram)
        else:
            token_query = self.create_ngrams(self.tokenizer.tokenize(query), n=self.n_gram)
        score_keyword = pd.Series([self.calc_jaccard_score(token_query, r) for r in token_reference], index=page_container["page_number"].to_list())
        score_position = pd.Series(linear_decay_vector, index=page_container["page_number"].to_list())
        score = (score_keyword + score_position).sort_values(ascending=False)
        return score

    def chunk(
            self, page_container, page_separator="\n\n",
            splitter_config=[
                {"type": "llm_text", "params": {"chunk_size": 3072, "chunk_overlap": 3072 // 4}},
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
                while cnt < self.max_trials:
                    try:
                        if self.llm_model_type == "openai":
                            completion = self.client.chat.completions.create(
                                model=self.llm_model_name,
                                messages=[
                                    {"role": "user", "content": self.prompt_template["user"].format(context=chunk_data.strip()).strip()},
                                ],
                                **self.sampling_params,
                                seed=self.seed,
                            )
                        # for vLLM
                        else:
                            completion = self.client.chat.completions.create(
                                model=self.llm_model_name,
                                messages=[
                                    {"role": "user", "content": self.prompt_template["user"].format(context=chunk_data.strip()).strip()},
                                ],
                                extra_body=self.sampling_params,
                                seed=self.seed,
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
                linear_decay_vector = self.create_linear_decay_vector(len(page_container), center_point=max(1, int(round(len(page_container) * ((chunk_id + 1) / len(chunks)), 0))))
                for chunk_id, cmpl in enumerate(completion.choices):
                    gened_data = cmpl.message.content
                    # searching page number
                    score = self.search_page_number(gened_data, page_container, linear_decay_vector)
                    output = {
                        "page_number": score.index[0],
                        "chunk_type": chunk_type,
                        "chunk_id": chunk_id,
                        "chunk_content": gened_data.strip(),       
                    }
                    yield output


class LLMBasedImageChunker():
    def __init__(
            self,
            llm_model_type="openai",
            llm_server_config={},
            llm_model_name="gpt-4o-mini",
            sampling_params={"max_completion_tokens": 1024, "n": 1, "temperature": 0.5, "top_p": 0.95},
            max_trials=5,
            prompt_template={
                "system": '',
                "user": """
이미지를 주제별로 나누어 한국어로 요약해 주세요.
"""
            },
            seed=42,
        ):
        self.llm_model_type = llm_model_type
        self.client = OpenAI(**llm_server_config)
        self.sampling_params = sampling_params
        self.llm_model_name = llm_model_name
        self.max_trials = max_trials
        self.prompt_template = prompt_template
        self.seed = seed
    def chunk(self, page_container, chunk_type="llm_image"):
        total_chunks = len(page_container)
        total_chunks *= self.sampling_params["n"]
        yield total_chunks

        for _, (_, row) in enumerate(tqdm(page_container.iterrows(), desc=f"create documents... ({chunk_type})", total=len(page_container))):
            # generation
            cnt = 0
            while cnt < self.max_trials:
                try:
                    if self.llm_model_type == "openai":
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
                            **self.sampling_params,
                            seed=self.seed,
                        )
                    # for vLLM
                    else:
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
                            seed=self.seed,
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
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                    "chunk_content": gened_data.strip(),  
                }
                yield output
