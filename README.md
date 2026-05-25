# Manga Text Extractor - One Piece French Edition

A specialized program for extracting French text from One Piece manga pages. This tool is optimized for manga panels, speech bubbles, and French language OCR.

## Features

- **French Language Support**: Optimized OCR configuration for French text with proper handling of accents and special characters
- **Manga-Specific Preprocessing**: Specialized image processing for manga pages
- **Speech Bubble Detection**: Automatic detection of speech bubbles and text regions
- **Text Region Classification**: Identifies different types of text regions (speech bubbles, narration boxes, sound effects, etc.)
- **Confidence Scoring**: Provides confidence levels for extracted text
- **Visualization**: Creates annotated images showing detected text regions
- **Batch Processing**: Process multiple manga pages at once
- **Debug Mode**: Generate intermediate images for troubleshooting

## Requirements

### System Dependencies

- **Tesseract OCR**: `sudo apt-get install tesseract-ocr`
- **French Language Data**: `sudo apt-get install tesseract-ocr-fra`
- **Python 3.7+**

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install opencv-python numpy pytesseract Pillow
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/breyting/One-piece-text-extractor.git
   cd One-piece-text-extractor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR with French support:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install tesseract-ocr tesseract-ocr-fra
   
   # On macOS
   brew install tesseract tesseract-lang
   ```

## Usage

### Basic Usage

Extract text from a manga page:

```bash
python manga_ocr.py 4.jpg
```

### Save Results to File

```bash
python manga_ocr.py 4.jpg --output result.txt
```

### Extract Full Text Only

```bash
python manga_ocr.py 4.jpg --full-text
```

### Create Visualization

```bash
python manga_ocr.py 4.jpg --visualize
```

### Debug Mode (with intermediate images)

```bash
python manga_ocr.py 4.jpg --debug
```

### Specify Tesseract Data Directory

```bash
python manga_ocr.py 4.jpg --tessdata /path/to/tessdata
```

### Run Examples

```bash
python example_usage.py
```

## Python API

### Basic Extraction

```python
from manga_ocr import MangaTextExtractor

# Create extractor
extractor = MangaTextExtractor()

# Extract full text
text = extractor.extract_full_page_text("manga_page.jpg")
print(text)

# Clean up
extractor.cleanup()
```

### Detailed Extraction with Regions

```python
from manga_ocr import MangaTextExtractor

# Create extractor
extractor = MangaTextExtractor()

# Extract detailed results
regions = extractor.extract_text("manga_page.jpg")

for region in regions:
    print(f"Type: {region.region_type.value}")
    print(f"Position: {region.bbox}")
    print(f"Confidence: {region.confidence:.1f}%")
    print(f"Text: {region.text}")
    print()

# Save results to file
extractor.save_results_to_file(regions, "results.txt")

# Create visualization
extractor.visualize_results("manga_page.jpg", regions, "visualization.jpg")

# Clean up
extractor.cleanup()
```

### Batch Processing

```python
from manga_ocr import MangaTextExtractor
import glob

# Create extractor
extractor = MangaTextExtractor()

# Process all manga pages in directory
for image_file in glob.glob("*.jpg"):
    text = extractor.extract_full_page_text(image_file)
    output_file = image_file.replace(".jpg", ".txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Processed {image_file} -> {output_file}")

extractor.cleanup()
```

## Text Region Types

The extractor classifies text regions into the following types:

- `SPEECH_BUBBLE`: Dialogue text in speech bubbles
- `THOUGHT_BUBBLE`: Character thoughts in thought bubbles
- `NARRATION_BOX`: Narrative text boxes
- `CHARACTER_NAME`: Character names
- `SOUND_EFFECT`: Sound effects (e.g., "BOOM", "CRASH")
- `PANEL_TEXT`: Text within manga panels
- `UNKNOWN`: Unclassified text regions

## French Language Optimization

The extractor includes several optimizations for French text:

- **Accent Handling**: Proper recognition of French accents (é, è, ê, ç, etc.)
- **Ligature Support**: Handling of French ligatures (œ, æ)
- **Typography Rules**: Correct spacing before punctuation (French style)
- **Common Corrections**: Automatic correction of common OCR errors in French
- **Language Model**: Uses French language model for better accuracy

## Image Preprocessing

The extractor applies several preprocessing steps to improve OCR accuracy:

1. **Contrast Enhancement**: Adaptive histogram equalization (CLAHE)
2. **Denoising**: Bilateral filtering to reduce noise while preserving edges
3. **Speech Bubble Detection**: Edge detection and contour analysis
4. **Text Region Detection**: MSER (Maximally Stable Extremal Regions)
5. **Region Merging**: Combines overlapping text regions
6. **Thresholding**: Otsu's thresholding for binarization
7. **Morphological Operations**: Clean up binary images

## Performance Tips

1. **Image Quality**: Higher resolution images generally yield better results
2. **Lighting**: Ensure good lighting when scanning manga pages
3. **Alignment**: Straight, non-skewed images work best
4. **Preprocessing**: Use the `--debug` flag to see intermediate results and adjust preprocessing as needed

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Install Tesseract OCR and ensure it's in your PATH
2. **French language data missing**: Install `tesseract-ocr-fra` package
3. **Low accuracy**: Try using higher quality images or enable debug mode to see preprocessing steps
4. **Memory issues**: Process one page at a time for large images

### Debug Mode

Enable debug mode to see intermediate processing steps:

```bash
python manga_ocr.py 4.jpg --debug
```

This will create a temporary directory with images showing:
- Original image
- Enhanced contrast
- Denoised image
- Detected speech bubbles
- Detected text regions
- Individual region crops

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- OpenCV: https://opencv.org/
- PyTesseract: https://github.com/madmaze/pytesseract

## Example Output

When you run the extractor on a One Piece manga page with French text, you'll get output like:

```
Processing manga page: 4.jpg

EXTRACTION RESULTS:
==================================================

Region 1 (speech_bubble):
  Position: (120, 450)
  Size: 320x180
  Confidence: 87.50
  Text: Luffy: Je vais devenir le Roi des Pirates!

Region 2 (speech_bubble):
  Position: (500, 300)
  Size: 280x120
  Confidence: 92.30
  Text: Zoro: Tu as encore oublié notre accord...

Region 3 (narration_box):
  Position: (100, 50)
  Size: 800x60
  Confidence: 88.70
  Text: Dans le vaste monde d'One Piece, l'aventure continue...
```

The extracted text will have proper French accents and formatting, ready for translation or analysis.