from functools import cache
from rapidocr import RapidOCR


@cache
def get_ocr_engine():
    return RapidOCR()


def extract_text(image_path: str) -> str:
    engine = get_ocr_engine()
    result = engine(image_path)   # get_ocr_engine()(image_path) 是两次调用叠在一起写
    return "\n".join(result.txts)