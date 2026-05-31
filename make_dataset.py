# !pip install google-generativeai python-dotenv tqdm datasets trl unsloth pymupdf
import re
import os
import json
import time
import concurrent.futures
from tqdm import tqdm
import fitz  # PyMuPDF
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

generation_config = {
    "response_mime_type": "application/json",
    "response_schema": {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "question": {"type": "STRING"},
                "answer": {"type": "STRING"},
                "complexity": {"type": "STRING"}
            },
            "required": ["question", "answer", "complexity"]
        }
    }
}

model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)

def chunk_pdfs_from_directory(directory_path, max_chars=1200):
    """
    Scans a directory for PDFs, extracts text efficiently using PyMuPDF,
    and splits the combined text into structured chunks.
    """
    all_chunks = []
    
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"The path '{directory_path}' does not exist. Verify the path configuration.")
        
    pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"Warning: No PDF files found in {directory_path}")
        return []
        
    print(f"Found {len(pdf_files)} PDF(s). Starting text extraction via PyMuPDF...")
    
    for pdf_file in pdf_files:
        full_path = os.path.join(directory_path, pdf_file)
        print(f"Processing: {pdf_file}")
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(full_path)
            file_text = ""
            
            for page in doc:
                text_content = page.get_text()
                if text_content:
                    file_text += text_content + "\n"
            
            doc.close()
            
            # Clean up white spaces and split by paragraph breaks
            file_text = re.sub(r'\n+', '\n', file_text)
            paragraphs = re.split(r'\n\s*\n', file_text)
            
            current_chunk = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) > max_chars and current_chunk:
                    all_chunks.append(current_chunk.strip())
                    current_chunk = para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
            
            if current_chunk:
                all_chunks.append(current_chunk.strip())
                
        except Exception as e:
            print(f"Failed to read file {pdf_file} due to: {e}")
            
    return all_chunks


pdf_folder_path = r"path_to_the_directory"


chunks = chunk_pdfs_from_directory(pdf_folder_path, max_chars=1200)

print(f"\n--- Processing Complete ---")
print(f"Total structured chunks prepared for Gemini: {len(chunks)}")

def is_high_quality(qa_pair):
    q = qa_pair.get("question", "").strip()
    a = qa_pair.get("answer", "").strip()
    
    if len(q) < 15 or len(a) < 20:
        return False
    if "i don't know" in a.lower() or "not mentioned" in a.lower():
        return False
    return True

def process_single_chunk(chunk):
    prompt = f"""
    You are an expert curriculum developer generating high-quality training datasets.
    Based on the text provided, generate exactly 4 QA pairs:
    - 1 Beginner question/answer
    - 1 Advanced question/answer
    - 1 Deep reasoning question/answer
    - 1 Real-world application scenario question/answer

    TEXT:
    {chunk}
    """
    try:
        response = model.generate_content(prompt)
        raw_data = json.loads(response.text)
        
        
        filtered_qa = [qa for qa in raw_data if is_high_quality(qa)]
        return filtered_qa
    except Exception as e:
        return []

dataset = []
MAX_WORKERS = 5 

if chunks:
    print(f"Starting concurrent generation over {len(chunks)} chunks...")
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_single_chunk, chunks), total=len(chunks)))

    for qa_list in results:
        for qa in qa_list:
            dataset.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful, accurate expert assistant."},
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]}
                ]
            })

    print(f"Generation Complete in {time.time() - start_time:.2f} seconds!")
    print(f"Total Validated QA Pairs Generated: {len(dataset)}")

    output_file = r"dataset\dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
else:
    print("No chunks available. Processing skipped.")