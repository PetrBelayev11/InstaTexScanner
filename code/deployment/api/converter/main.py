from latex_converter import LatexConverter

# Initialize the converter (enables segmentation by default)
converter = LatexConverter(enable_segmentation=True)

# Convert a document image to LaTeX
result = converter.convert(
    image_path="test_image3.png",
    out_dir="output",
    segment_document=True  # Enable document segmentation
)

print(f"Document type: {result['type']}")
print(f"Extracted text: {result['text']}")
print(f"LaTeX file: {result['latex_file']}")
print(f"Found {result['segments_count']} segments")