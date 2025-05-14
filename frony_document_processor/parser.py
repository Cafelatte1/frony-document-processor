import os
import io
import base64
import pandas as pd
import pdfplumber
from operator import itemgetter
from PIL import Image
import uuid
import platform

class ParserTXT():
    def __init__(self):
        pass

    def parse(self, file: io.BytesIO | str):
        try:
            if isinstance(file, str):
                with open(file, 'r', encoding='utf-8') as f:
                    file = f.read()
            else:
                file.seek(0)
                file = file.read().decode('utf-8')
            page_container = [{
                'page_number': 1,
                'page_content': file.strip()
            }]
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None
    
class ParserPDF():
    def __init__(self, x_tolerance_ratio=0.15, keep_blank_chars=True):
        self.extract_words_params = {
            "x_tolerance_ratio": x_tolerance_ratio,
            "keep_blank_chars": keep_blank_chars,
        }
    
    @staticmethod
    def is_overlap(box1, box2):
        # box1의 우측 상단이 box2의 좌측 하단보다 왼쪽에 있거나
        # box1의 좌측 하단이 box2의 우측 상단보다 오른쪽에 있으면 겹치지 않음
        if (box1[2] < box2[0]) or (box2[2] < box1[0]):
            return False
        # box1의 상단이 box2의 하단보다 아래에 있거나
        # box1의 하단이 box2의 상단보다 위에 있으면 겹치지 않음
        if (box1[3] < box2[1]) or (box2[3] < box1[1]):
            return False
        # 위의 조건을 모두 만족하지 않으면 겹침
        return True

    @staticmethod
    def extract_table(page, table):
        croppage = page.crop((0, max(0, table.bbox[1]), page.width, min(table.bbox[3], page.height)))
        edgel = sorted(croppage.horizontal_edges, key=itemgetter("x0"))[0]
        edger = sorted(croppage.horizontal_edges, key=itemgetter("x1"))[-1]
        table = croppage.extract_table({"vertical_strategy": "lines", "explicit_vertical_lines": [edgel["x0"], edger["x1"]]})
        return [[]] if table is None else table

    def get_page_data(self, page):
        page_data = {"text": [], "table": []}
        # create text data
        for word in page.extract_words(**self.extract_words_params):
            page_data["text"].append({
                "content": word["text"],
                "coord": (word["x0"], word["top"], word["x1"], word["bottom"]),
            })
        # create table data
        for table in page.find_tables():
            data = self.extract_table(page, table)
            if len(data) >= 2:
                tb = pd.DataFrame(data[1:], columns=data[0])
                tb.columns = tb.columns.fillna("")
                tb = tb.fillna("")
                tb = tb.drop(tb.columns[tb.nunique() == 1], axis=1)
                page_data["table"].append({
                    "content": tb.to_markdown(),
                    "coord": table.bbox,
                })
        # create dataframe
        page_data = {k: pd.DataFrame(v) for k, v in page_data.items()}
        # post-processing for text
        if len(page_data["text"]) > 0:
            page_data["text"] = page_data["text"].groupby(page_data["text"]["coord"].apply(lambda x: (x[1] + x[3]) / 2).astype("int"), sort=False, as_index=False).agg({"content": " ".join, "coord": "first"})
        # post-processing for table
        if len(page_data["table"]) > 0:
            page_data["table"] = page_data["table"].groupby(page_data["table"]["coord"].apply(lambda x: (x[1] + x[3]) / 2).astype("int"), sort=False, as_index=False).agg({"content": " ".join, "coord": "first"})
            page_data["table"]["content"] = page_data["table"]["content"].apply(lambda x: f"\n{x}\n")
            for tb_coord in page_data["table"]["coord"]:
                mask = page_data["text"]["coord"].apply(lambda x: not self.is_overlap(x, tb_coord))
                page_data["text"] = page_data["text"][mask].reset_index(drop=True)
        df = pd.concat([pd.DataFrame(page_data["text"]), pd.DataFrame(page_data["table"])], axis=0).reset_index(drop=True)
        df = df.iloc[df["coord"].apply(lambda x: x[1]).argsort()].reset_index(drop=True)
        return "\n".join(df["content"])

    def parse(self, file: io.BytesIO | str):
        try:
            doc = pdfplumber.open(file)
            page_container = []
            for page_number, page in enumerate(doc.pages):
                page_container.append({
                    "page_number": page_number + 1,
                    "page_content": self.get_page_data(page).strip(),
                })
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None

class ParserPPTX():
    def __init__(self, cache_dir="./.cache/parser-pptx/", resolution=300):
        self.cache_dir = cache_dir
        self.resolution = resolution

    @staticmethod
    def encode_image(buffer):
        buffer.seek(0)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return data

    def pptx_to_pdf(self, file: io.BytesIO):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # 임시 PPTX 파일 생성
        file_id = uuid.uuid4().hex
        src_path = os.path.join(self.cache_dir, f"{file_id}.pptx")
        dst_path = os.path.join(self.cache_dir, f"{file_id}.pdf")
        with open(src_path, 'wb') as f:
            file.seek(0)
            f.write(file.read())
        try:
            libreoffice_path = r'"C:\Program Files\LibreOffice\program\soffice.exe"' if platform.system() == "Windows" else "soffice"
            os.system(f"{libreoffice_path} --headless --convert-to pdf --outdir {self.cache_dir} {src_path}")
            print(f"complete conversion from PPTX to PDF / output_path={dst_path}")
            with open(dst_path, 'rb') as f:
                file = io.BytesIO(f.read())
            return file
        except Exception as e:
            print(f"error in conversion -> {e}")
            return None
        finally:
            # 임시 파일 정리
            if os.path.exists(src_path):
                os.remove(src_path)
            if os.path.exists(dst_path):
                os.remove(dst_path)

    def parse(self, file: io.BytesIO | str):
        try:
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    file = io.BytesIO(f.read())
            doc = pdfplumber.open(self.pptx_to_pdf(file))
            page_container = []
            for page_number, page in enumerate(doc.pages):
                buffer = io.BytesIO()
                img = page.to_image(resolution=self.resolution).original
                img.save(buffer, format="png")
                page_container.append({
                    "page_number": page_number + 1,
                    "page_content": self.encode_image(buffer),
                })
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None
        finally:
            if 'buffer' in locals():
                buffer.close()

class ParserPDFImage():
    def __init__(self, resolution=300):
        self.resolution = resolution

    @staticmethod
    def encode_image(buffer):
        buffer.seek(0)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return data
    
    def parse(self, file: io.BytesIO | str):
        try:
            doc = pdfplumber.open(file)
            page_container = []
            for page_number, page in enumerate(doc.pages):
                buffer = io.BytesIO()
                img = page.to_image(resolution=self.resolution).original
                img.save(buffer, format="png")
                page_container.append({
                    "page_number": page_number + 1,
                    "page_content": self.encode_image(buffer),
                })
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None
        finally:
            if 'buffer' in locals():
                buffer.close()
    
class ParserImage():
    def __init__(self):
        pass

    @staticmethod
    def encode_image(buffer):
        buffer.seek(0)
        data = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return data
    
    def parse(self, file: io.BytesIO | str):
        try:
            img = Image.open(file)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            page_container = [{
                "page_number": 1,
                "page_content": self.encode_image(buffer),
            }]
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None
        finally:
            if 'buffer' in locals():
                buffer.close()

class ParserDOCX():
    def __init__(self, x_tolerance_ratio=0.15, keep_blank_chars=True, cache_dir="./.cache/parser-docx/"):
        self.cache_dir = cache_dir
        self.extract_words_params = {
            "x_tolerance_ratio": x_tolerance_ratio,
            "keep_blank_chars": keep_blank_chars,
        }

    @staticmethod
    def is_overlap(box1, box2):
        # box1의 우측 상단이 box2의 좌측 하단보다 왼쪽에 있거나
        # box1의 좌측 하단이 box2의 우측 상단보다 오른쪽에 있으면 겹치지 않음
        if (box1[2] < box2[0]) or (box2[2] < box1[0]):
            return False
        # box1의 상단이 box2의 하단보다 아래에 있거나
        # box1의 하단이 box2의 상단보다 위에 있으면 겹치지 않음
        if (box1[3] < box2[1]) or (box2[3] < box1[1]):
            return False
        # 위의 조건을 모두 만족하지 않으면 겹침
        return True

    @staticmethod
    def extract_table(page, table):
        croppage = page.crop((0, max(0, table.bbox[1]), page.width, min(table.bbox[3], page.height)))
        edgel = sorted(croppage.horizontal_edges, key=itemgetter("x0"))[0]
        edger = sorted(croppage.horizontal_edges, key=itemgetter("x1"))[-1]
        table = croppage.extract_table({"vertical_strategy": "lines", "explicit_vertical_lines": [edgel["x0"], edger["x1"]]})
        return [[]] if table is None else table

    def get_page_data(self, page):
        page_data = {"text": [], "table": []}
        # create text data
        for word in page.extract_words(**self.extract_words_params):
            page_data["text"].append({
                "content": word["text"],
                "coord": (word["x0"], word["top"], word["x1"], word["bottom"]),
            })
        # create table data
        for table in page.find_tables():
            data = self.extract_table(page, table)
            if len(data) >= 2:
                tb = pd.DataFrame(data[1:], columns=data[0])
                tb.columns = tb.columns.fillna("")
                tb = tb.fillna("")
                tb = tb.drop(tb.columns[tb.nunique() == 1], axis=1)
                page_data["table"].append({
                    "content": tb.to_markdown(),
                    "coord": table.bbox,
                })
        # create dataframe
        page_data = {k: pd.DataFrame(v) for k, v in page_data.items()}
        # post-processing for text
        if len(page_data["text"]) > 0:
            page_data["text"] = page_data["text"].groupby(page_data["text"]["coord"].apply(lambda x: (x[1] + x[3]) / 2).astype("int"), sort=False, as_index=False).agg({"content": " ".join, "coord": "first"})
        # post-processing for table
        if len(page_data["table"]) > 0:
            page_data["table"] = page_data["table"].groupby(page_data["table"]["coord"].apply(lambda x: (x[1] + x[3]) / 2).astype("int"), sort=False, as_index=False).agg({"content": " ".join, "coord": "first"})
            page_data["table"]["content"] = page_data["table"]["content"].apply(lambda x: f"\n{x}\n")
            for tb_coord in page_data["table"]["coord"]:
                mask = page_data["text"]["coord"].apply(lambda x: not self.is_overlap(x, tb_coord))
                page_data["text"] = page_data["text"][mask].reset_index(drop=True)
        df = pd.concat([pd.DataFrame(page_data["text"]), pd.DataFrame(page_data["table"])], axis=0).reset_index(drop=True)
        df = df.iloc[df["coord"].apply(lambda x: x[1]).argsort()].reset_index(drop=True)
        return "\n".join(df["content"])
    
    def docx_to_pdf(self, file: io.BytesIO):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # 임시 PPTX 파일 생성
        file_id = uuid.uuid4().hex
        src_path = os.path.join(self.cache_dir, f"{file_id}.pptx")
        dst_path = os.path.join(self.cache_dir, f"{file_id}.pdf")
        with open(src_path, 'wb') as f:
            file.seek(0)
            f.write(file.read())
        try:
            libreoffice_path = r'"C:\Program Files\LibreOffice\program\soffice.exe"' if platform.system() == "Windows" else "soffice"
            os.system(f"{libreoffice_path} --headless --convert-to pdf --outdir {self.cache_dir} {src_path}")
            print(f"complete conversion from PPTX to PDF / output_path={dst_path}")
            with open(dst_path, 'rb') as f:
                file = io.BytesIO(f.read())
            return file
        except Exception as e:
            print(f"error in conversion -> {e}")
            return None
        finally:
            # 임시 파일 정리
            if os.path.exists(src_path):
                os.remove(src_path)
            if os.path.exists(dst_path):
                os.remove(dst_path)

    def parse(self, file: io.BytesIO | str):
        try:
            if isinstance(file, str):
                with open(file, 'rb') as f:
                    file = io.BytesIO(f.read())
            doc = pdfplumber.open(self.docx_to_pdf(file))
            page_container = []
            for page_number, page in enumerate(doc.pages):
                page_container.append({
                    "page_number": page_number + 1,
                    "page_content": self.get_page_data(page).strip(),
                })
            page_container = pd.DataFrame(page_container)
            return page_container
        except Exception as e:
            print(f"error in parsing -> {e}")
            return None
