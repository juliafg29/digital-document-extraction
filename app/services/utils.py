import os
from pdf2image import convert_from_path
from PIL import Image
from PIL.Image import DecompressionBombError
import cv2
import numpy as np

import logging
from paddleocr import PaddleOCR

def pdf_para_jpg(caminho_pdf, dpi=300):
    """
    Converte cada página de um PDF em imagens JPG.
    Ignora PDFs problemáticos (ex: muito grandes).
    """

    # Arquivos com muita resolução não serão considerados na convesão para imagem para proteção do computador
    try:
        paginas = convert_from_path(caminho_pdf, dpi=dpi)
    except DecompressionBombError:
        print(f"[IGNORADO] {os.path.basename(caminho_pdf)} → imagem muito grande (possível decompression bomb)")
        return
    except Exception as e:
        print(f"[ERRO] {os.path.basename(caminho_pdf)} → {e}")
        return

    # Adapta nome do arquivo
    #nome_pdf = os.path.splitext(os.path.basename(caminho_pdf))[0]
    todas_paginas = []

    # Para cada pagina do PDF é gerado uma imagem
    for i in enumerate(paginas):
        imagem_cv = cv2.cvtColor(np.array(paginas[i]), cv2.COLOR_RGB2BGR)
        todas_paginas.append(imagem_cv)

    return todas_paginas

def compose_paddle_ocr():
    logging.getLogger('ppocr').setLevel(logging.WARNING)

    ocr = PaddleOCR(use_angle_cls=True, lang='pt')

def gather_results(results:list):

    if not results:
        return None

    # Começa usando o primeiro resultado como base
    resultado_final = results[0].copy()

    extracao_final = resultado_final.get("extracao", {})
    campos_finais = extracao_final.get("campos", {})
    confianca_final = extracao_final.get("confianca", {})

    for resultado in results[1:]:

        extracao = resultado.get("extracao", {})
        campos = extracao.get("campos", {})
        confianca = extracao.get("confianca", {})

        for campo, valor in campos.items():

            valor_atual = campos_finais.get(campo)

            # Verifica se o valor atual está vazio
            vazio = (
                valor_atual is None
                or valor_atual == ""
                or str(valor_atual).strip().lower() == "none"
            )

            # Se estiver vazio e o novo valor existir, substitui
            if vazio and valor not in [None, "", "None"]:

                campos_finais[campo] = valor

                if campo in confianca:
                    confianca_final[campo] = confianca[campo]

    # Atualiza estrutura final
    resultado_final["extracao"]["campos"] = campos_finais
    resultado_final["extracao"]["confianca"] = confianca_final

    return resultado_final