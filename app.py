from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
from PyPDF2 import PdfReader
import io
import re

# --- Configuración principal ---
app = FastAPI(title="Falcon Chat PDF", version="6.0")

# --- Configuración del frontend ---
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Cargar modelo principal (TinyLlama o Falcon) ---
try:
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype="auto",
        device_map="auto"
    )
    print("✅ Modelo Falcon cargado correctamente.")
except Exception as e:
    raise RuntimeError(f"❌ Error al cargar el modelo Falcon: {str(e)}")

# --- Cargar modelo de traducción ---
try:
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
    print("✅ Modelo de traducción cargado correctamente.")
except Exception:
    translator = None
    print("⚠️ No se pudo cargar el modelo de traducción. Traducción deshabilitada.")


# --- Función: Extraer texto del PDF ---
def extract_text_from_pdf(file_data: UploadFile) -> str:
    """
    Extrae texto limpio y sin caracteres extraños del PDF.
    Devuelve el texto o un mensaje de advertencia si falla.
    """
    try:
        content = file_data.file.read()
        if not content:
            return "⚠️ El archivo PDF está vacío o no se pudo leer."
        reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += re.sub(r'\s+', ' ', page_text.strip()) + " "
        text = text.strip()

        if not text:
            return "⚠️ No se pudo extraer texto del PDF (puede estar escaneado o protegido)."

        # Limitar tamaño del texto para evitar desbordes de tokens
        if len(text) > 6000:
            text = text[:6000].rsplit(" ", 1)[0] + "..."
        return text

    except Exception as e:
        return f"❌ Error al procesar el PDF: {str(e)}"


# --- Página principal ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Carga la interfaz principal."""
    return templates.TemplateResponse("index.html", {"request": request})


# --- Procesamiento del texto extraído del PDF ---
@app.post("/process")
async def process_pdf(file: UploadFile = File(...), action: str = Form(...)):
    text = extract_text_from_pdf(file)
    if not text or text.startswith(("⚠️", "❌")):
        return JSONResponse({"response": text})

    # Prompts según la acción seleccionada
    prompts = {
        "tema": (
            "Analiza el siguiente texto y explica de manera clara y completa de qué trata. "
            "Describe el propósito, contexto y mensaje principal:\n\n"
        ),
        "resumen": (
            "Genera un resumen amplio, coherente y bien estructurado del siguiente texto. "
            "Incluye las ideas principales y evita repetir frases textuales:\n\n"
        ),
        "conclusion": (
            "Redacta una conclusión desarrollada y reflexiva sobre el siguiente texto, "
            "mencionando sus implicaciones y proyecciones futuras:\n\n"
        ),
        "keywords": (
            "Enumera las 10 palabras o conceptos clave más importantes del siguiente texto. "
            "Deben reflejar los temas principales y su relevancia:\n\n"
        ),
        "recomendaciones": (
            "Propón recomendaciones, posibles mejoras o aplicaciones prácticas derivadas del siguiente texto. "
            "Incluye sugerencias para investigaciones futuras o mejoras de implementación:\n\n"
        ),
        "traduccion": (
            "Traduce el siguiente texto al inglés con precisión, manteniendo el sentido original:\n\n"
        )
    }

    # Encabezados de las respuestas
    intro = {
        "tema": "🧩 Explicación general del documento:",
        "resumen": "📝 Resumen detallado:",
        "conclusion": "🔚 Conclusión desarrollada:",
        "keywords": "🔑 Palabras clave identificadas:",
        "recomendaciones": "💡 Recomendaciones y aplicaciones sugeridas:",
        "traduccion": "🌎 Traducción al inglés:"
    }

    prompt = prompts.get(action, "Analiza el siguiente texto:") + text

    try:
        # Caso especial: traducción directa
        if action == "traduccion" and translator:
            translated = translator(text, max_length=2000)[0]["translation_text"]
            return JSONResponse({
                "response": f"{intro[action]} {translated}",
                "text": None
            })

        # Generación estándar con control de errores
        output = generator(
            prompt,
            max_new_tokens=700,
            temperature=0.65,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )[0].get("generated_text", "")

        # Limpieza del texto generado
        clean = output.replace(prompt, "").strip()
        clean = re.sub(r"\s{2,}", " ", clean)
        clean = re.sub(r"\n+", " ", clean)
        clean = re.sub(r"^.*?:", "", clean, 1).strip()

        if not clean:
            clean = "⚠️ No se pudo generar una respuesta válida."

        return JSONResponse({
            "response": f"{intro.get(action, 'Resultado:')} {clean}",
            "text": text if action == "tema" else None
        })

    except Exception as e:
        return JSONResponse({
            "response": f"❌ Error al generar respuesta: {str(e)}"
        })
