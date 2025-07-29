# PDF Document Structure Extraction Pipeline

This project is a hybrid pipeline for extracting the structure and outline (titles and headings) from PDF documents. It leverages *LightGBM* for feature-based classification and a *Transformer model* for sequence classification to achieve robust predictions. The pipeline processes PDF files, extracts text blocks, and classifies them into structural elements like Title, H1–H4 headings, and Body.

Another point of note is that this model has multi-lingual support.
---

## *Key Components*

### *1. Models*
- *LightGBM Model* (lgbm_filter.model): Predicts structural classes from a set of engineered features.
- *Transformer Model* (HuggingFace): Provides contextual classification using text-classification pipeline.
- *Class Labels* are stored in lgbm_classes.npy.

### *2. Feature Engineering*
The following features are extracted from each text block:
- Font properties: font_size, is_bold, is_italic, relative_font_size
- Position: x_position_normalized, y_position, is_centered
- Structural indicators: space_below, line_height, span_count, starts_with_numbering
- Text properties: text_length, is_all_caps

### *3. PDF Parsing*
- Uses *PyMuPDF (fitz)* to parse PDFs.
- group_nearby_blocks() merges text blocks based on proximity and font similarity.
- process_page_wrapper() extracts text blocks per page and computes features.

---

## *Pipeline Design*

### *Parallel & Sequential Pipelines*
1. *Parallel* (run_pipeline): Uses ProcessPoolExecutor for faster inference during final submission.
2. *Sequential* (run_pipeline_sequential): Processes pages in a loop for stable results during model training.

### *Hybrid Classification*
1. Predict with *LightGBM* and filter out "Body" class blocks.
2. For candidate blocks, run Transformer-based classification:
   - Accept Transformer prediction if confidence ≥ 0.75.
   - Otherwise, fallback to LightGBM prediction.

### *Output*
The pipeline returns:
- *Title:* The first block predicted as "Title" or fallback to the first text block.
- *Outline:* Hierarchically sorted list of headings with page numbers.

Example output:
```json
{
    "title": "Document Title",
    "outline": [
        {"level": "H1", "text": "Introduction", "page": 1},
        {"level": "H2", "text": "Background", "page": 2}
    ]
}

Execution Instructions for Round 1A:
This guide explains how to build the Docker image and run the container to process a set of PDF files, generating a separate JSON outline for each one.

Prerequisites:
Docker Desktop must be installed and running on your system.

Step 1: Folder Arrangement
Before running the commands, ensure your folders are structured correctly. The input folder must be in the same directory as your Dockerfile.

<project_root>/
├── input/
│   └── (Place all your PDFs for analysis here, e.g., file01.pdf, file02.pdf)
│
├── output/
│   └── (This folder should be empty; results will appear here)
│
└── Dockerfile
Place all the PDF files you want to process directly inside the input folder.

Step 2: Build the Docker Image
Open your terminal (PowerShell, Command Prompt, etc.) in the project's root directory and run the following command to build the Docker image.

docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

This command reads the Dockerfile, installs all necessary dependencies, and packages your application into a self-contained image named mysolutionname:somerandomidentifier.

Step 3: Run the Docker Container
Once the image is built successfully, run the following command to process the documents.

On Windows (PowerShell):

docker run --rm -v "${pwd}/input:/app/input:ro" -v "${pwd}/output:/app/output" --network none mysolutionname:somerandomidentifier

On Linux or macOS:

docker run --rm -v "$(pwd)/input:/app/input:ro" -v "$(pwd)/output:/app/output" --network none mysolutionname:somerandomidentifier

This command starts a new container from your image.

The -v flags create a link between your local input and output folders and the /app/input and /app/output folders inside the container.

The --network none flag ensures the container runs completely offline, as required.

After the command finishes, the output folder will contain a separate filename.json for each filename.pdf you provided in the input folder.
