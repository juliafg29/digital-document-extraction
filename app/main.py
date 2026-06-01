from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from tempfile import NamedTemporaryFile
import shutil
import os

from app.services.document_workflow import document_workflow

app = FastAPI(
    title="Serviço de Validação e Extração de Documentos",
    description="API para validação, extração de dados e geração XML de documentos pessoais digitalizados.",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "mensagem": "API de processamento de documentos ativa."
    }


@app.post("/documentos/analisar")
async def analisar_documento(
    arquivo: UploadFile = File(...)
):
    """
    Endpoint responsável por:
    - receber um PDF
    - validar o documento
    - extrair dados
    - gerar XML
    """

    # Verifica extensão
    if not arquivo.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Somente PDF é permitido.")

    conteudo_inicial = await arquivo.read(5)
    if conteudo_inicial != b"%PDF-":
        raise HTTPException(status_code=400, detail="Arquivo PDF inválido.")

    await arquivo.seek(0)

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    conteudo = await arquivo.read()
    if len(conteudo) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Arquivo muito grande.")

    try:
        # Salva o arquivo temporariamente
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            shutil.copyfileobj(arquivo.file, temp_file)
            caminho_temporario = temp_file.name

        # Processa documento
        MIN_SCORE = 0.5
        resultado_com_xml = document_workflow(caminho_temporario, MIN_SCORE)

        print("O resultado foi: " , resultado_com_xml["xml"])
        
        return JSONResponse(content={
            "status": "sucesso",
            "resultado": resultado_com_xml
        })

    except Exception as erro:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar documento: {str(erro)}"
        )

    finally:
        # Remove arquivo temporário
        if caminho_temporario and os.path.exists(caminho_temporario):
            os.remove(caminho_temporario)