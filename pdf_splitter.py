import os
import fitz  # PyMuPDF

def split_pdfs_to_images(input_folder: str, output_folder: str = "TrainImages"):
    os.makedirs(output_folder, exist_ok=True)
    image_counter = 0

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_folder, filename)
            document = fitz.open(pdf_path)

            for page_number in range(len(document)):
                page = document.load_page(page_number)
                pix = page.get_pixmap()
                output_filename = f"ATOM_DOC_{image_counter:03}.png"
                output_path = os.path.join(output_folder, output_filename)
                pix.save(output_path)
                image_counter += 1

    print(f"Processed {image_counter} images from PDFs in '{input_folder}'")

# Example usage
split_pdfs_to_images("GeoDocs")
