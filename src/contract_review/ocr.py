from functools import cache
from rapidocr import RapidOCR
import pymupdf


@cache
def get_ocr_engine():
    return RapidOCR()


def extract_text(image_path: str) -> str:
    engine = get_ocr_engine()
    result = engine(image_path)   # get_ocr_engine()(image_path) 是两次调用叠在一起写
    return "\n".join(result.txts)


def extract_text_from_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    parts = []
    for page in doc:
        pix = page.get_pixmap(dpi=200)
        result = get_ocr_engine()(pix.tobytes("png"))
        parts.extend(result.txts)
    doc.close()
    return "\n".join(parts)