#!/usr/bin/env python3
"""
Example usage of the Manga Text Extractor for One Piece French manga pages.

This script demonstrates how to use the manga_ocr.py module to extract text
from One Piece manga pages with French text.
"""

import os
import sys
from manga_ocr import MangaTextExtractor, check_dependencies


def example_basic_extraction():
    """Basic example: Extract text from a manga page"""
    print("=" * 60)
    print("BASIC EXTRACTION EXAMPLE")
    print("=" * 60)
    
    # Create extractor
    extractor = MangaTextExtractor(debug=False)
    
    try:
        # Extract text from the example image
        image_path = "4.jpg"
        if not os.path.exists(image_path):
            print(f"Example image {image_path} not found!")
            return
        
        print(f"Processing {image_path}...")
        
        # Extract all text as a single string
        full_text = extractor.extract_full_page_text(image_path)
        
        print("\nEXTRACTED TEXT:")
        print("-" * 40)
        print(full_text)
        print("-" * 40)
        
    finally:
        extractor.cleanup()


def example_detailed_extraction():
    """Detailed example: Extract text with region information"""
    print("\n" + "=" * 60)
    print("DETAILED EXTRACTION EXAMPLE")
    print("=" * 60)
    
    # Create extractor
    extractor = MangaTextExtractor(debug=False)
    
    try:
        image_path = "4.jpg"
        if not os.path.exists(image_path):
            print(f"Example image {image_path} not found!")
            return
        
        print(f"Processing {image_path}...")
        
        # Extract detailed results with region information
        results = extractor.extract_text(image_path)
        
        print(f"\nFound {len(results)} text regions:")
        print("-" * 40)
        
        for i, region in enumerate(results, 1):
            print(f"\nRegion {i}:")
            print(f"  Type: {region.region_type.value}")
            print(f"  Position: ({region.bbox[0]}, {region.bbox[1]})")
            print(f"  Size: {region.bbox[2]}x{region.bbox[3]} pixels")
            print(f"  Confidence: {region.confidence:.1f}%")
            print(f"  Text: {region.text}")
        
        # Save results to file
        output_file = "extraction_results.txt"
        extractor.save_results_to_file(results, output_file)
        print(f"\nResults saved to {output_file}")
        
        # Create visualization
        visualization_file = "visualization.jpg"
        extractor.visualize_results(image_path, results, visualization_file)
        print(f"Visualization saved to {visualization_file}")
        
    finally:
        extractor.cleanup()


def example_with_debug():
    """Example with debug mode enabled (creates intermediate images)"""
    print("\n" + "=" * 60)
    print("DEBUG MODE EXAMPLE")
    print("=" * 60)
    
    # Create extractor with debug mode
    extractor = MangaTextExtractor(debug=True)
    
    try:
        image_path = "4.jpg"
        if not os.path.exists(image_path):
            print(f"Example image {image_path} not found!")
            return
        
        print(f"Processing {image_path} with debug mode...")
        print("(Intermediate images will be saved to a temporary directory)")
        
        # Extract text
        results = extractor.extract_text(image_path)
        
        print(f"\nExtracted {len(results)} text regions with debug information.")
        
    finally:
        extractor.cleanup()


def example_batch_processing():
    """Example: Process multiple manga pages"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Create extractor
    extractor = MangaTextExtractor(debug=False)
    
    try:
        # Find all JPEG images in the current directory
        import glob
        image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")
        
        if not image_files:
            print("No image files found in the current directory.")
            return
        
        print(f"Found {len(image_files)} manga pages to process:")
        
        for image_file in image_files:
            print(f"\nProcessing {image_file}...")
            
            try:
                # Extract text
                full_text = extractor.extract_full_page_text(image_file)
                
                # Save to text file
                output_file = os.path.splitext(image_file)[0] + ".txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                
                print(f"  Text extracted and saved to {output_file}")
                
            except Exception as e:
                print(f"  Error processing {image_file}: {e}")
        
        print(f"\nBatch processing completed!")
        
    finally:
        extractor.cleanup()


def main():
    """Run all examples"""
    print("Manga Text Extractor - Example Usage")
    print("Optimized for French One Piece manga pages")
    print()
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install the required dependencies and try again.")
        sys.exit(1)
    
    # Run examples
    example_basic_extraction()
    example_detailed_extraction()
    example_with_debug()
    example_batch_processing()
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED")
    print("=" * 60)
    print("\nYou can now use the manga_ocr.py script directly:")
    print("  python manga_ocr.py 4.jpg --output result.txt")
    print("  python manga_ocr.py 4.jpg --visualize")
    print("  python manga_ocr.py 4.jpg --full-text")
    print("  python manga_ocr.py 4.jpg --debug")


if __name__ == "__main__":
    main()