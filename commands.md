

Swap in your real documents. 
Drop .md/.txt files into data/raw and re-run 

python -m src.ingestion.index. 


Run the API:

uvicorn src.api.main:app --reload

