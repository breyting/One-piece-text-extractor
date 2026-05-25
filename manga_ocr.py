#!/usr/bin/env python3
"""
Manga Text Extractor - Optimized for French text and One Piece manga panels

This program extracts text from manga pages with specialized preprocessing
for speech bubbles, text panels, and French language support.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import os
import sys
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
from enum import Enum
import tempfile
import subprocess


class TextRegionType(Enum):
    """Types of text regions in manga"""
    SPEECH_BUBBLE = "speech_bubble"
    THOUGHT_BUBBLE = "thought_bubble"
    NARRATION_BOX = "narration_box"
    CHARACTER_NAME = "character_name"
    SOUND_EFFECT = "sound_effect"
    PANEL_TEXT = "panel_text"
    UNKNOWN = "unknown"


@dataclass
class TextRegion:
    """Represents a detected text region in a manga page"""
    text: str
    region_type: TextRegionType
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    language: str = "fra"
    
    def __str__(self):
        return f"TextRegion(text='{self.text}', type={self.region_type.value}, bbox={self.bbox}, confidence={self.confidence:.2f})"


class MangaTextExtractor:
    """
    Main class for extracting text from manga pages.
    Optimized for French text and One Piece style panels.
    """
    
    def __init__(self, 
                 tessdata_prefix: Optional[str] = None,
                 use_gpu: bool = False,
                 debug: bool = False):
        """
        Initialize the manga text extractor.
        
        Args:
            tessdata_prefix: Path to Tesseract data directory
            use_gpu: Whether to use GPU acceleration (if available)
            debug: Enable debug mode with intermediate image outputs
        """
        self.debug = debug
        self.tessdata_prefix = tessdata_prefix
        self.use_gpu = use_gpu
        
        # Configure Tesseract
        self._configure_tesseract()
        
        # French language configuration
        self.french_config = self._get_french_ocr_config()
        
        # Create temporary directory for debug images
        if self.debug:
            self.temp_dir = tempfile.mkdtemp(prefix="manga_ocr_debug_")
            print(f"Debug images will be saved to: {self.temp_dir}")
        
    def _configure_tesseract(self):
        """Configure Tesseract OCR with optimal settings for manga"""
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except EnvironmentError:
            print("Tesseract OCR is not installed. Please install it:")
            print("  sudo apt-get install tesseract-ocr")
            print("  sudo apt-get install tesseract-ocr-fra")
            print("  pip install pytesseract")
            sys.exit(1)
        
        # Set Tesseract path if specified
        if self.tessdata_prefix:
            os.environ['TESSDATA_PREFIX'] = self.tessdata_prefix
        
        # Configure Tesseract to use French language
        self.tesseract_config = r'--oem 3 --psm 6 -l fra+eng'
        if self.use_gpu:
            self.tesseract_config += ' --tessdata-dir /usr/share/tesseract-ocr/4.00/tessdata'
    
    def _get_french_ocr_config(self) -> Dict:
        """Get optimal OCR configuration for French manga text"""
        return {
            'lang': 'fra',
            'preserve_interword_spaces': True,
            'oem': 3,  # LSTM OCR engine
            'psm': 6,  # Assume a single uniform block of text
            'dpi': 300,
            'char_whitelist': '',  # Allow all characters
            'load_system_dawg': True,
            'load_freq_dawg': True,
            'textord_tabfind_multiply_by_aspect_ratio': 1.5,
        }
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess the manga image for OCR.
        
        Args:
            image_path: Path to the manga image file
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to RGB (OpenCV loads as BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.debug:
            cv2.imwrite(f"{self.temp_dir}/01_original.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return image
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast for better text detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        if self.debug:
            cv2.imwrite(f"{self.temp_dir}/02_enhanced_contrast.jpg", enhanced)
        
        return enhanced
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges"""
        # Apply bilateral filter to reduce noise while keeping edges sharp
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        
        if self.debug:
            cv2.imwrite(f"{self.temp_dir}/03_denoised.jpg", denoised)
        
        return denoised
    
    def _detect_speech_bubbles(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect speech bubbles in manga panels.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of bounding boxes (x, y, w, h) for detected speech bubbles
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        speech_bubbles = []
        
        for contour in contours:
            # Filter by area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum area for a speech bubble
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # Speech bubbles are typically wider than tall
            if aspect_ratio > 1.5 and aspect_ratio < 10:
                # Expand the bounding box a bit to capture text near edges
                x = max(0, x - 10)
                y = max(0, y - 10)
                w = min(image.shape[1] - x, w + 20)
                h = min(image.shape[0] - y, h + 20)
                
                speech_bubbles.append((x, y, w, h))
        
        if self.debug:
            # Draw detected speech bubbles for debugging
            debug_img = image.copy()
            for (x, y, w, h) in speech_bubbles:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(f"{self.temp_dir}/04_speech_bubbles.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        return speech_bubbles
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential text regions using MSER (Maximally Stable Extremal Regions)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create MSER detector
        mser = cv2.MSER_create()
        
        # Detect regions
        regions, _ = mser.detectRegions(gray)
        
        # Filter and merge regions
        text_regions = []
        
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter by size (text regions should be reasonably sized)
            if w < 20 or h < 20:
                continue
            if w > image.shape[1] // 2 or h > image.shape[0] // 2:
                continue
            
            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.1 or aspect_ratio > 20:
                continue
            
            text_regions.append((x, y, w, h))
        
        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions(text_regions)
        
        if self.debug:
            # Draw detected text regions for debugging
            debug_img = image.copy()
            for (x, y, w, h) in merged_regions:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.imwrite(f"{self.temp_dir}/05_text_regions.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        return merged_regions
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]], 
                                  overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes"""
        if not regions:
            return []
        
        # Convert to list of rectangles
        rects = [(x, y, x + w, y + h) for x, y, w, h in regions]
        
        # Simple merging algorithm
        merged = []
        for rect in rects:
            x1, y1, x2, y2 = rect
            
            # Check if this rectangle overlaps with any existing merged rectangle
            found_overlap = False
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                # Calculate overlap area
                overlap_x1 = max(x1, mx1)
                overlap_y1 = max(y1, my1)
                overlap_x2 = min(x2, mx2)
                overlap_y2 = min(y2, my2)
                
                overlap_width = max(0, overlap_x2 - overlap_x1)
                overlap_height = max(0, overlap_y2 - overlap_y1)
                overlap_area = overlap_width * overlap_height
                
                # Calculate area of current rectangle
                rect_area = (x2 - x1) * (y2 - y1)
                
                # If overlap is significant, merge
                if overlap_area / rect_area > overlap_threshold:
                    # Merge the rectangles
                    new_x1 = min(x1, mx1)
                    new_y1 = min(y1, my1)
                    new_x2 = max(x2, mx2)
                    new_y2 = max(y2, my2)
                    
                    merged[i] = (new_x1, new_y1, new_x2, new_y2)
                    found_overlap = True
                    break
            
            if not found_overlap:
                merged.append(rect)
        
        # Convert back to (x, y, w, h) format
        return [(x1, y1, x2 - x1, y2 - y1) for x1, y1, x2, y2 in merged]
    
    def _extract_text_from_region(self, image: np.ndarray, 
                                 region: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Extract text from a specific region using OCR.
        
        Args:
            image: Input image
            region: Bounding box (x, y, w, h)
            
        Returns:
            Tuple of (extracted_text, confidence)
        """
        x, y, w, h = region
        
        # Crop the region
        cropped = image[y:y+h, x:x+w]
        
        if self.debug:
            cv2.imwrite(f"{self.temp_dir}/region_{x}_{y}_{w}_{h}.jpg", 
                       cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
        
        # Preprocess the cropped region
        processed = self._preprocess_region(cropped)
        
        # Use PIL for better OCR results
        pil_image = Image.fromarray(processed)
        
        try:
            # Extract text with Tesseract
            text = pytesseract.image_to_string(
                pil_image, 
                config=self.tesseract_config,
                lang='fra'
            )
            
            # Get confidence (this is a simplified approach)
            # For more accurate confidence, we'd need to use pytesseract.image_to_data
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.tesseract_config,
                lang='fra',
                output_type=pytesseract.Output.DICT
            )
            
            confidence = np.mean([float(c) for c in data['conf'] if c != '-1']) if data['conf'] else 0.0
            
            # Clean up the text
            text = self._clean_text(text)
            
            return text, confidence
            
        except Exception as e:
            print(f"Error extracting text from region {region}: {e}")
            return "", 0.0
    
    def _preprocess_region(self, region: np.ndarray) -> np.ndarray:
        """Preprocess a cropped region for OCR"""
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Invert back
        cleaned = cv2.bitwise_not(cleaned)
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text by removing unwanted characters and formatting"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Fix common OCR errors for French
        text = self._fix_french_ocr_errors(text)
        
        return text.strip()
    
    def _fix_french_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors specific to French text"""
        # Common French character corrections
        corrections = {
            'à': ['a', 'à', 'à'],  # à (a with grave)
            'â': ['a', 'â'],       # â (a with circumflex)
            'ä': ['a', 'ä'],       # ä (a with umlaut)
            'ç': ['c', 'ç'],       # ç (c with cedilla)
            'é': ['e', 'é'],       # é (e with acute)
            'è': ['e', 'è'],       # è (e with grave)
            'ê': ['e', 'ê'],       # ê (e with circumflex)
            'ë': ['e', 'ë'],       # ë (e with umlaut)
            'î': ['i', 'î'],       # î (i with circumflex)
            'ï': ['i', 'ï'],       # ï (i with umlaut)
            'ô': ['o', 'ô'],       # ô (o with circumflex)
            'ö': ['o', 'ö'],       # ö (o with umlaut)
            'ù': ['u', 'ù'],       # ù (u with grave)
            'û': ['u', 'û'],       # û (u with circumflex)
            'ü': ['u', 'ü'],       # ü (u with umlaut)
            'ÿ': ['y', 'ÿ'],       # ÿ (y with umlaut)
            'œ': ['oe'],           # œ ligature
            'æ': ['ae'],           # æ ligature
            '«': ['<<', '"'],      # French quotation marks
            '»': ['>>', '"'],      # French quotation marks
            '–': ['-'],            # En dash to hyphen
            '—': ['-'],            # Em dash to hyphen
            '…': ['...'],          # Ellipsis
        }
        
        # Apply corrections
        for correct_char, possible_errors in corrections.items():
            for error in possible_errors:
                text = text.replace(error, correct_char)
        
        # Fix common OCR misreadings
        text = text.replace('l\'', 'l\'')  # Preserve French contractions
        text = text.replace('d\'', 'd\'')
        text = text.replace('qu\'', 'qu\'')
        
        # Fix spaces before punctuation (French typography)
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace(' ;', ';')
        text = text.replace(' :', ':')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        
        return text
    
    def _classify_region_type(self, region: Tuple[int, int, int, int], 
                             image_shape: Tuple[int, int]) -> TextRegionType:
        """
        Classify the type of text region based on position and characteristics.
        
        Args:
            region: Bounding box (x, y, w, h)
            image_shape: Shape of the original image (height, width)
            
        Returns:
            TextRegionType classification
        """
        x, y, w, h = region
        img_height, img_width = image_shape
        
        # Calculate position ratios
        x_ratio = x / img_width
        y_ratio = y / img_height
        w_ratio = w / img_width
        h_ratio = h / img_height
        aspect_ratio = w / h
        
        # Classification rules
        
        # Sound effects are usually large, irregular, and in the action areas
        if h_ratio > 0.1 and w_ratio > 0.1 and aspect_ratio > 2:
            return TextRegionType.SOUND_EFFECT
        
        # Character names are usually at the top of speech bubbles or panels
        if y_ratio < 0.1 and h_ratio < 0.05:
            return TextRegionType.CHARACTER_NAME
        
        # Narration boxes are usually at the top or bottom of the page
        if (y_ratio < 0.05 or y_ratio > 0.95) and w_ratio > 0.5:
            return TextRegionType.NARRATION_BOX
        
        # Thought bubbles are typically cloud-shaped and in the middle
        # (This would need more sophisticated shape analysis)
        
        # Default to speech bubble for most text regions
        return TextRegionType.SPEECH_BUBBLE
    
    def extract_text(self, image_path: str) -> List[TextRegion]:
        """
        Extract text from a manga page.
        
        Args:
            image_path: Path to the manga image file
            
        Returns:
            List of TextRegion objects containing extracted text and metadata
        """
        print(f"Processing manga page: {image_path}")
        
        # Load and preprocess image
        image = self._preprocess_image(image_path)
        
        # Detect speech bubbles
        speech_bubbles = self._detect_speech_bubbles(image)
        
        # Detect general text regions
        text_regions = self._detect_text_regions(image)
        
        # Combine regions (prioritize speech bubbles)
        all_regions = speech_bubbles + text_regions
        
        # Remove duplicates
        unique_regions = []
        seen = set()
        for region in all_regions:
            region_key = (region[0] // 10, region[1] // 10)  # Round to nearest 10px to avoid duplicates
            if region_key not in seen:
                seen.add(region_key)
                unique_regions.append(region)
        
        # Extract text from each region
        results = []
        image_shape = image.shape[:2]  # (height, width)
        
        for i, region in enumerate(unique_regions):
            print(f"Processing region {i+1}/{len(unique_regions)}: {region}")
            
            text, confidence = self._extract_text_from_region(image, region)
            
            if text.strip():  # Only keep regions with actual text
                region_type = self._classify_region_type(region, image_shape)
                
                text_region = TextRegion(
                    text=text,
                    region_type=region_type,
                    bbox=region,
                    confidence=confidence,
                    language="fra"
                )
                
                results.append(text_region)
                print(f"  Extracted: '{text}' (confidence: {confidence:.2f})")
        
        # Sort regions by position (top to bottom, left to right)
        results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        return results
    
    def extract_full_page_text(self, image_path: str) -> str:
        """
        Extract all text from a manga page as a single string.
        
        Args:
            image_path: Path to the manga image file
            
        Returns:
            Combined text from all regions
        """
        regions = self.extract_text(image_path)
        
        # Combine text from all regions
        all_text = []
        for region in regions:
            if region.text.strip():
                all_text.append(region.text)
        
        return '\n\n'.join(all_text)
    
    def save_results_to_file(self, results: List[TextRegion], 
                           output_path: str) -> None:
        """
        Save extraction results to a file.
        
        Args:
            results: List of TextRegion objects
            output_path: Path to save the results
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Manga Text Extraction Results\n")
            f.write("=" * 50 + "\n\n")
            
            for i, region in enumerate(results, 1):
                f.write(f"Region {i} ({region.region_type.value}):\n")
                f.write(f"Position: ({region.bbox[0]}, {region.bbox[1]})\n")
                f.write(f"Size: {region.bbox[2]}x{region.bbox[3]}\n")
                f.write(f"Confidence: {region.confidence:.2f}\n")
                f.write(f"Text:\n{region.text}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Results saved to: {output_path}")
    
    def visualize_results(self, image_path: str, 
                         results: List[TextRegion], 
                         output_path: str) -> None:
        """
        Create a visualization of the extraction results.
        
        Args:
            image_path: Path to the original image
            results: List of TextRegion objects
            output_path: Path to save the visualization
        """
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Draw bounding boxes and labels
        for i, region in enumerate(results, 1):
            x, y, w, h = region.bbox
            
            # Choose color based on region type
            color_map = {
                TextRegionType.SPEECH_BUBBLE: (0, 255, 0),    # Green
                TextRegionType.THOUGHT_BUBBLE: (0, 255, 255), # Yellow
                TextRegionType.NARRATION_BOX: (255, 0, 0),    # Blue
                TextRegionType.CHARACTER_NAME: (0, 0, 255),   # Red
                TextRegionType.SOUND_EFFECT: (255, 0, 255),   # Magenta
                TextRegionType.PANEL_TEXT: (255, 255, 0),     # Cyan
                TextRegionType.UNKNOWN: (128, 128, 128),      # Gray
            }
            
            color = color_map.get(region.region_type, (128, 128, 128))
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with region number and type
            label = f"{i}: {region.region_type.value}"
            cv2.putText(image, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw confidence
            conf_label = f"{region.confidence:.1f}%"
            cv2.putText(image, conf_label, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save the visualization
        cv2.imwrite(output_path, image)
        print(f"Visualization saved to: {output_path}")
    
    def cleanup(self):
        """Clean up temporary files"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")


def check_dependencies() -> bool:
    """Check if all required dependencies are installed"""
    # Map package names to their import names
    required_packages = [
        ('opencv-python', 'cv2'),
        ('numpy', 'numpy'),
        ('pytesseract', 'pytesseract'),
        ('Pillow', 'PIL')
    ]
    missing = []
    
    for package, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    # Check for Tesseract OCR
    try:
        pytesseract.get_tesseract_version()
    except EnvironmentError:
        print("Tesseract OCR is not installed.")
        print("Install it with:")
        print("  sudo apt-get install tesseract-ocr")
        print("  sudo apt-get install tesseract-ocr-fra")
        return False
    
    return True


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Extract text from One Piece manga pages (optimized for French text)'
    )
    
    parser.add_argument('image_path', 
                       help='Path to the manga image file')
    parser.add_argument('--output', '-o', 
                       help='Output file path for extracted text')
    parser.add_argument('--visualize', '-v', 
                       action='store_true',
                       help='Create visualization of detected regions')
    parser.add_argument('--debug', '-d', 
                       action='store_true',
                       help='Enable debug mode with intermediate images')
    parser.add_argument('--tessdata', 
                       help='Path to Tesseract data directory')
    parser.add_argument('--full-text', '-t',
                       action='store_true',
                       help='Output full text instead of detailed results')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create extractor
    extractor = MangaTextExtractor(
        tessdata_prefix=args.tessdata,
        debug=args.debug
    )
    
    try:
        if args.full_text:
            # Extract full text
            text = extractor.extract_full_page_text(args.image_path)
            print("\n" + "="*50)
            print("EXTRACTED TEXT:")
            print("="*50)
            print(text)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"\nFull text saved to: {args.output}")
        else:
            # Extract detailed results
            results = extractor.extract_text(args.image_path)
            
            print("\n" + "="*50)
            print("EXTRACTION RESULTS:")
            print("="*50)
            
            for i, region in enumerate(results, 1):
                print(f"\nRegion {i} ({region.region_type.value}):")
                print(f"  Position: ({region.bbox[0]}, {region.bbox[1]})")
                print(f"  Size: {region.bbox[2]}x{region.bbox[3]}")
                print(f"  Confidence: {region.confidence:.2f}")
                print(f"  Text: {region.text}")
            
            # Save results if output path is specified
            if args.output:
                extractor.save_results_to_file(results, args.output)
            
            # Create visualization if requested
            if args.visualize:
                viz_path = args.output or "visualization.jpg"
                if not viz_path.endswith('.jpg'):
                    viz_path = viz_path + '.viz.jpg'
                extractor.visualize_results(args.image_path, results, viz_path)
        
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()