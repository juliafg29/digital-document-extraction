# digital-document-extraction


docker build -t api-documentos .
docker run -p 8000:8000 api-documentos


- A API não armazena documentos enviados.
- Arquivos temporários são removidos após o processamento.
- Logs não registram dados pessoais.
- O acesso ao endpoint exige chave de API.
- O serviço aceita apenas arquivos PDF.