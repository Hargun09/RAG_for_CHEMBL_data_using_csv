# A chatbot for Female disorders

This project builds a dataset connecting **drugs → targets → genes → diseases (MONDO)**, using data from the ChEMBL API and UniProt, and prepares it for retrieval (RAG) using a FAISS index.


### ACCESS ON THE FOLLOWING LINK 
https://ragforfemaledisorders.streamlit.app/

## Pipeline Stages & Files

### 1. Drug & Indication Data (ChEMBL)
- `drug_indications` — Raw drug–disease indication data pulled from ChEMBL (`drug_indication` API).
- `chembl_activity_targets` — Drug activity and target data from ChEMBL.

### 2. Target / UniProt Processing
- `clean_uniprot_ids` — Cleaned list of UniProt IDs extracted from the ChEMBL target data.
- `uniprot_enriched` / `uniprot_enriched - Copy` — UniProt IDs enriched with additional protein/target metadata.

### 3. Gene Mapping
- `uniprot_to_gene_final` — Final mapping of UniProt IDs to gene names.
- `uniprot_to_gene_for_female` — Gene mapping filtered/adjusted for female-specific context.

### 4. Disease Ontology (MONDO)
- `mondo.owl` — Raw MONDO ontology file.
- `mondo_data` — Extracted MONDO disease data.
- `mondo_with_hierarchy` — MONDO diseases mapped with their parent/child hierarchy (disease classes).
- `mondos` / `mondos_final` — Processed and finalized MONDO disease mapping tables.

### 5. Merged / Final Datasets
- `data_names`, `datadata` — Intermediate/supporting data files.
- `data` (zip) — Archived data folder.
- `final_final`, `final_final - Copy`, `check2` — Final merged datasets combining drug, target, gene, and disease information.

### 6. Retrieval Index (RAG)
- `index.faiss`, `index.pkl`, `index_pkl` (zip) — Vector embeddings and index built from the final dataset, used for retrieval-augmented querying (e.g. via LlamaIndex).

