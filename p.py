from xhtml2pdf import pisa             # import python module

# Define your data
source_html = open('./a.html', 'r')

output_filename = "test.pdf"

# Utility function
def convert_html_to_pdf(source_html, output_filename):
    with open(output_filename, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(source_html, dest=pdf_file)
        
    return not pisa_status.err

# Main program
if __name__ == "__main__":
    pisa.showLogging()
    convert_html_to_pdf(source_html, output_filename)

    source_html.close()