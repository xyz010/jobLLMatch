import os

import fitz

# Input directory containing PDF files
# input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/SWE_junior_le_2years_pdf"
# input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/SWE_senior_ge_6years_pdf"
# input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/SWE_mid_ge_3years_le_5years_pdf"
# input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/frontend_mid_pdf"

# bucket data
input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/swe_big_resume_bucket_pdf"
# input_directory = "/Users/andreasloutzidis/Downloads/indeed_data/ml_other_big_resume_bucket_pdf"


# Output directory to store JSON files
# output_directory = "/Users/andreasloutzidis/Downloads/indeed_data/SWE_junior_le_2years_txt"
# output_directory = "/Users/andreasloutzidis/Downloads/indeed_data/SWE_senior_ge_6years_txt"
# output_directory = "/Users/andreasloutzidis/Downloads/indeed_data/frontend_mid_txt"
output_directory = "/Users/andreasloutzidis/Downloads/indeed_data/swe_ds_ml_resume_bucket_txt"


# Function to extract and save "work experience" or "professional experience" section
def extract_text_from_pdf(pdf_path, output_dir):
    # Extract the filename from the provided path
    pdf_filename = os.path.basename(pdf_path)
    pdf_filename_no_extension = os.path.splitext(pdf_filename)[0]

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Extract the text from the PDF file
    text = ""
    for page in pdf_document:
        text += page.get_text()

    # Save the text to a .txt file with the same filename
    output_file = os.path.join(output_dir, f"{pdf_filename_no_extension}.txt")
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(text)

    # Close the PDF document
    pdf_document.close()



final_dataset_path = "/Users/andreasloutzidis/Downloads/indeed_data/final_dataset"
final_resume_path = os.path.join(final_dataset_path, "resume", "resume_raw")
final_jd_path = os.path.join(final_dataset_path, "job_description")

swe_pdf_path = os.path.join(final_resume_path, "swe_big_resume_bucket_pdf")
ml_other_pdf_path = os.path.join(final_resume_path, "ml_other_big_resume_bucket_pdf")
ds_pdf_path = os.path.join(final_resume_path, "ds_big_resume_bucket_pdf")
jd_pdf_path = os.path.join(final_jd_path, "job_descr_swe_big_bucket_pdf")


input_directory = ds_pdf_path
output_directory = os.path.join(final_resume_path, "resume_unity_txt")
# Iterate through each PDF file in the input directory
for pdf_filename in os.listdir(input_directory):
    if pdf_filename.endswith(".pdf"):
        pdf_file_path = os.path.join(input_directory, pdf_filename)
        extract_text_from_pdf(pdf_path=pdf_file_path, output_dir=output_directory)
        print(f"Converted {pdf_filename} to {pdf_file_path}")

print("Conversion completed.")
