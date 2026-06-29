import pypdf
import docx
from pathlib import Path

class DocumentParser:
    def parse(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext == '.docx':
            return self._parse_docx(file_path)
        else:
            return open(file_path, 'r', errors='ignore').read()

    def _parse_pdf(self, path: str) -> str:
        reader = pypdf.PdfReader(path)
        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
        return text

    def _parse_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return '\n'.join(p.text for p in doc.paragraphs if p.text.strip())