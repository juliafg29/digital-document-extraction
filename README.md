# digital-document-extraction

## Running with Docker

Build the Docker image:

```bash
docker build -t api-documentos .
```

Run the container:

```bash
docker run -p 8000:8000 api-documentos
```

Recommendations for digitized documents:
- Avoid colored backgrounds
- Avoid shadows or markings on top of the document image
- Preferably send the entire document on a single page
- Only upload PDF files

Important security alerts: 
- The API does not store submitted documents.
- Temporary files are removed after processing.
- Logs do not record personal data.
- Access to the endpoint requires an API key.
- The service only accepts PDF files.


```
