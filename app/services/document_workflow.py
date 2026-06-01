import app.services.utils as utils
from app.services.extract_digitalized_data import processar_documento
from app.services.xml_utils import gerar_xml

def document_workflow(input_file_path, MIN_SCORE = 0.5):

    # Pdf to Image
    all_image_pages = utils.convert_from_path(input_file_path, dpi = 300)

    #STEPS FOR DIGITALIZED DOCUMENTS
    # 1. Compose Paddle OCR
    utils.compose_paddle_ocr()

    # 2. Extract data with PadddleOCR
    all_results = []
    final_result = []

    for image in all_image_pages:
        result = processar_documento(image, MIN_SCORE)

        tipo_doc = result.get("tipo_documento", {})
        print(f"  Tipo: {tipo_doc.get('tipo')}\n")

        extracao = result.get("extracao", {})
        campos = extracao.get("campos", {})
        confianca = extracao.get("confianca", {})

        print("\nCAMPOS_EXTRAIDOS\n")
        for campo, valor in campos.items():
            print(f"  {campo}: {valor} [{confianca[campo]}]\n")

        all_results.append(result)

    # Choose the best option for all fields
    if len(all_results) > 1:
        final_result = utils.gather_results(all_results)
    else:
        final_result = all_results

    print("\nRESULTADO FINAL\n")

    for campo, valor in final_result["extracao"]["campos"].items():
        conf = final_result["extracao"]["confianca"].get(campo)
        print(f"{campo}: {valor} [{conf}]")


    #STEPS FOR DIGITAL CARTEIRA NACIONAL DE HABILITAÇÃO ONLY
    # {...}
    #dados, texto_ocr = extract_ecnh(input_path,lang ="por+eng")

    # XML compose
    final_result_with_xml = gerar_xml(final_result, caminho_pdf=input_file_path)

    return final_result_with_xml