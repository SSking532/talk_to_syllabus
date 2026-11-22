
from pypdf import PdfReader


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)

    extracted_pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text()

        if extracted and extracted.strip():
            extracted_pages.append(
                f"\n--- PAGE {page_number} ---\n{extracted.strip()}"
            )

    text = "\n".join(extracted_pages)

    if not text.strip():
        raise ValueError(
            "No readable text found in this PDF. "
            "The PDF may contain scanned images instead of selectable text."
        )

    return text
