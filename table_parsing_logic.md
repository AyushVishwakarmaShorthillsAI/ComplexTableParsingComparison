# Table Parsing & Enhancement Logic Documentation

This document outlines the two-pass approach used to extract and enhance table data from PDFs, transforming it into a format optimized for RAG (Retrieval-Augmented Generation) and semantic search.

## Overview

The system avoids the common pitfalls of raw table extraction (which often results in broken formatting or lost context) by using a combination of structural analysis and multimodal LLM vision.

### Key Tools & Packages

| Component | Tool/Package | Purpose |
| :--- | :--- | :--- |
| **PDF Processing** | `PyMuPDF` (`fitz`) | PDF-to-image rendering and metadata extraction. |
| **Table Detection** | `Docling` (IBM) | Structural analysis to identify pages containing tables. |
| **Vision/Extraction** | `Gemini 2.x Flash` | Converting table images into descriptive natural language. |
| **Text Processing** | `NLTK` | Sentence-aware tokenization for clean chunking. |
| **Image Handling** | `Pillow` (`PIL`) | Image resizing and pre-processing for API limits. |

---

## Technical Workflow

The logic is split into two distinct phases to ensure efficiency and high-quality results.

### Phase 1: Structural Detection (The "First Pass")
*Implemented in `DocumentChunker.py`*

1.  **Conversion**: The PDF is processed using `Docling`'s `DocumentConverter`.
2.  **Structural Mapping**: Docling identifies various elements (Headings, Paragraphs, Tables, Figures).
3.  **Table Page Indexing**: The code iterates through `doc.tables` and records the `page_no` for every table found.
4.  **Selective Filtering**: During initial text chunking, items labeled as `table` are explicitly **ignored**. This prevents low-quality raw text dumps of tables from polluting the vector store.
5.  **Metadata Storage**: The identified "table pages" are stored in a `pdf_details` map within the `chunks.json` output.

### Phase 2: Semantic Enhancement (The "Second Pass")
*Implemented in `TableEnhancer.py`*

1.  **Rendering**: For every page flagged in Phase 1, the system renders that specific page into a high-resolution PNG (300 DPI) using `PyMuPDF`.
2.  **Vision Extraction**: The image is sent to Gemini with a specialized system prompt.
    *   **The Prompt Strategy**: Instead of asking for Markdown or CSV, Gemini is instructed to "Extract and describe the table content in a format optimized for semantic search".
    *   **Structure**: The output must include Table Title, Purpose, Structure, and most importantly, **Table Data as natural language paragraphs**.
3.  **Refining**:
    *   **Group Related Data**: Gemini describes rows and columns in complete sentences (e.g., "The revenue for Q1 2023 was $5M, representing a 10% increase...").
    *   **Context Retention**: It includes footnotes and key findings in the prose.
4.  **Chunking**: The resulting descriptive text is split into chunks (~1024 characters) with sentence-level overlap using `NLTK` to ensure no data point is lost at a chunk boundary.
5.  **Re-indexing**: These "Enhanced Table Chunks" are appended to the original `chunks.json`, enriching the dataset with high-quality, searchable table data.

---

## Replication Guide for External Apps

To replicate this logic, follow these steps:

### 1. Requirements
```bash
pip install docling pymupdf google-genai nltk pillow python-dotenv
```

### 2. Detection (Pseudo-code)
```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("document.pdf")
doc = result.document

table_pages = set()
for table in doc.tables:
    table_pages.add(table.prov[0].page_no)
```

### 3. Rendering (Pseudo-code)
```python
import fitz

doc = fitz.open("document.pdf")
page = doc.load_page(page_num - 1)
pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
pix.save(f"page_{page_num}.png")
```

### 4. Gemini Prompt Construction
Use this exact prompt for the best RAG performance:
> "You are a research paper assistant. This page contains a table. Extract and describe the table content in a format optimized for semantic search and RAG. 
> 1. Table Title/Caption. 
> 2. Table Purpose. 
> 3. Table Structure (columns/rows). 
> 4. Table Data: Present the data in natural language paragraphs (NOT as a raw table). Group related data, use complete sentences, include all values. 
> 5. Key Findings. 
> 6. Context (footnotes/notes)."

---

## Critical Considerations

> [!IMPORTANT]
> **Why Semantic Prose instead of Markdown Tables?**
> Standard embedding models struggle with the sparse, structural nature of Markdown/CSV tables. By converting tables into natural language paragraphs, the data gains "semantic density," making it much easier for vector search to find relevant rows based on natural language queries.

> [!WARNING]
> **Rate Limiting**
> Using a Vision model on every page is expensive and slow. The "Two Pass" logic (Docling first, Gemini second) is critical to only process pages that actually contain tables.
