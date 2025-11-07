from PyPDF2 import PdfReader
import re
import io

def extract_text_from_pdf(file_data):
    """
    Extrae texto legible y limpio desde un archivo PDF (bytes o UploadFile).
    Devuelve texto plano o un mensaje de error amigable.
    Compatible con el backend Falcon Chat PDF.
    """
    try:
        # Permitir tanto UploadFile como bytes puros
        if hasattr(file_data, "file"):
            content = file_data.file.read()
        elif isinstance(file_data, (bytes, bytearray)):
            content = file_data
        else:
            return "⚠️ Tipo de archivo no soportado."

        # Validar si está vacío
        if not content:
            return "⚠️ El archivo PDF está vacío o no se pudo leer."

        reader = PdfReader(io.BytesIO(content))
        if not reader.pages:
            return "⚠️ El PDF no contiene páginas legibles."

        text_parts = []

        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_text = re.sub(r'\s+', ' ', page_text).strip()
                if page_text:
                    text_parts.append(page_text)
                else:
                    text_parts.append(f"[⚠️ Página {i} sin texto legible]")
            except Exception:
                text_parts.append(f"[⚠️ Error al leer la página {i}]")

        # Unir y limpiar texto final
        text = " ".join(text_parts)
        text = re.sub(r'\s{2,}', ' ', text).strip()

        # Verificación final: evitar archivos escaneados sin texto
        if not text or all(seg.startswith("[⚠️") for seg in text_parts):
            return "⚠️ No se pudo extraer texto del PDF (puede estar escaneado o protegido)."

        # Limitar tamaño del texto (para evitar saturar el modelo)
        if len(text) > 6000:
            text = text[:6000].rsplit(" ", 1)[0] + "..."

        return text

    except Exception as e:
        return f"❌ Error al procesar el PDF: {str(e)}"
