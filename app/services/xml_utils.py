import os
import base64
import xml.etree.ElementTree as ET
from xml.dom import minidom


def gerar_xml(final_result, caminho_pdf=None):
    """
    Gera XML estruturado e adiciona ao dicionário final_result.

    Opcionalmente incorpora o PDF original em Base64.
    """

    campos = final_result.get("extracao", {}).get("campos", {})
    confianca = final_result.get("extracao", {}).get("confianca", {})
    tipo_doc = final_result.get("tipo_documento", {}).get("tipo")

    root = ET.Element("documento")

    # Tipo do documento
    tipo = ET.SubElement(root, "tipo_documento")
    tipo.text = tipo_doc if tipo_doc else ""

    # Dados extraídos
    dados = ET.SubElement(root, "dados_extraidos")

    for campo, valor in campos.items():

        item = ET.SubElement(dados, campo)

        valor_limpo = ""

        if valor not in [None, "", "None"]:
            valor_limpo = str(valor)

        item.text = valor_limpo

        # atributo de confiança
        if campo in confianca and confianca[campo] is not None:
            item.set("confianca", str(confianca[campo]))

    # PDF original
    if caminho_pdf and os.path.exists(caminho_pdf):

        pdf_element = ET.SubElement(root, "arquivo_original")

        nome_arquivo = ET.SubElement(pdf_element, "nome_arquivo")
        nome_arquivo.text = os.path.basename(caminho_pdf)

        formato = ET.SubElement(pdf_element, "formato")
        formato.text = "application/pdf"

        with open(caminho_pdf, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        conteudo = ET.SubElement(pdf_element, "conteudo_base64")
        conteudo.text = pdf_base64

    # XML formatado
    xml_bytes = ET.tostring(root, encoding="utf-8")

    xml_formatado = minidom.parseString(xml_bytes).toprettyxml(
        indent="  "
    )

    # adiciona XML ao resultado final
    final_result["xml"] = xml_formatado

    return final_result