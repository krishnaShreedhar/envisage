import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ImageAnalysis:
    """Data structure to hold image analysis results"""
    filename: str
    dimensions: Tuple[int, int]
    brightness_mean: float
    brightness_std: float
    rgb_means: Tuple[float, float, float]
    rgb_stds: Tuple[float, float, float]
    dominant_colors: List[Tuple[int, int, int]]
    composition_score: float
    aurora_intensity: float


class NorthernLightsProcessor:
    """Main class for processing Northern Lights images"""

    def __init__(self, input_folder: str, output_folder: str = "processed_images"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)

        # Create subdirectories
        (self.output_folder / "analysis").mkdir(exist_ok=True)
        (self.output_folder / "cleaned").mkdir(exist_ok=True)
        (self.output_folder / "visualizations").mkdir(exist_ok=True)

    def load_image(self, image_path: str) -> np.ndarray:
        """Load image and convert to RGB"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def analyze_composition(self, image: np.ndarray) -> Dict:
        """Analyze image composition using rule of thirds and other metrics"""
        height, width = image.shape[:2]

        # Convert to grayscale for composition analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Rule of thirds analysis
        third_h, third_w = height // 3, width // 3
        thirds_points = [
            (third_w, third_h), (2 * third_w, third_h),
            (third_w, 2 * third_h), (2 * third_w, 2 * third_h)
        ]

        # Calculate interest at rule of thirds points
        interest_score = 0
        for x, y in thirds_points:
            # Use gradient magnitude as interest measure
            roi = gray[max(0, y - 20):min(height, y + 20),
                  max(0, x - 20):min(width, x + 20)]
            if roi.size > 0:
                grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
                interest_score += np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2))

        # Horizon detection (for landscape composition)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        horizon_score = 0
        if lines is not None:
            horizontal_lines = []
            for rho, theta in lines[:, 0]:
                if abs(theta - np.pi / 2) < np.pi / 6:  # Near horizontal lines
                    horizontal_lines.append((rho, theta))
            if horizontal_lines:
                horizon_score = len(horizontal_lines) / len(lines)

        return {
            'rule_of_thirds_score': interest_score / 4,
            'horizon_score': horizon_score,
            'overall_composition_score': (interest_score / 4 + horizon_score) / 2
        }

    def analyze_rgb_intensities(self, image: np.ndarray) -> Dict:
        """Analyze RGB channel intensities and statistics"""
        r_channel = image[:, :, 0]
        g_channel = image[:, :, 1]
        b_channel = image[:, :, 2]

        # Calculate statistics for each channel
        stats = {
            'red': {
                'mean': float(np.mean(r_channel)),
                'std': float(np.std(r_channel)),
                'median': float(np.median(r_channel)),
                'percentile_95': float(np.percentile(r_channel, 95)),
                'percentile_5': float(np.percentile(r_channel, 5))
            },
            'green': {
                'mean': float(np.mean(g_channel)),
                'std': float(np.std(g_channel)),
                'median': float(np.median(g_channel)),
                'percentile_95': float(np.percentile(g_channel, 95)),
                'percentile_5': float(np.percentile(g_channel, 5))
            },
            'blue': {
                'mean': float(np.mean(b_channel)),
                'std': float(np.std(b_channel)),
                'median': float(np.median(b_channel)),
                'percentile_95': float(np.percentile(b_channel, 95)),
                'percentile_5': float(np.percentile(b_channel, 5))
            },
            'overall': {
                'brightness_mean': float(np.mean(image)),
                'brightness_std': float(np.std(image)),
            }
        }

        return stats

    def create_color_histogram(self, image: np.ndarray, save_path: str = None) -> Dict:
        """Create and optionally save color histograms"""
        # Calculate histograms
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])

        # Create visualization
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(hist_r, color='red', alpha=0.7)
        plt.title('Red Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        plt.plot(hist_g, color='green', alpha=0.7)
        plt.title('Green Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 3)
        plt.plot(hist_b, color='blue', alpha=0.7)
        plt.title('Blue Channel Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            'red_hist': hist_r.flatten().tolist(),
            'green_hist': hist_g.flatten().tolist(),
            'blue_hist': hist_b.flatten().tolist()
        }

    def detect_aurora_intensity(self, image: np.ndarray) -> float:
        """Detect aurora intensity based on green channel dominance and brightness"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Aurora typically appears in green/blue-green hues
        # HSV ranges for aurora colors (green to blue-green)
        aurora_mask1 = cv2.inRange(hsv, (40, 50, 50), (80, 255, 255))  # Green range
        aurora_mask2 = cv2.inRange(hsv, (80, 50, 50), (120, 255, 255))  # Blue-green range

        aurora_mask = cv2.bitwise_or(aurora_mask1, aurora_mask2)

        # Calculate aurora intensity
        aurora_pixels = cv2.countNonZero(aurora_mask)
        total_pixels = image.shape[0] * image.shape[1]
        aurora_ratio = aurora_pixels / total_pixels

        # Get average brightness in aurora regions
        if aurora_pixels > 0:
            aurora_brightness = np.mean(image[aurora_mask > 0])
            intensity = aurora_ratio * (aurora_brightness / 255)
        else:
            intensity = 0.0

        return float(intensity)

    def get_dominant_colors(self, image: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using K-means clustering"""
        # Reshape image to be a list of pixels
        pixels = image.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Apply K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to integers and return as list of tuples
        centers = np.uint8(centers)
        dominant_colors = [tuple(map(int, color)) for color in centers]

        return dominant_colors

    def clean_image(self, image: np.ndarray, noise_reduction: bool = True,
                    enhance_contrast: bool = True, sharpen: bool = True) -> np.ndarray:
        """Clean and enhance the image"""
        # Convert to PIL for easier processing
        pil_image = Image.fromarray(image)

        # Noise reduction using bilateral filter
        if noise_reduction:
            cv_image = cv2.bilateralFilter(image, 9, 75, 75)
            pil_image = Image.fromarray(cv_image)

        # Contrast enhancement
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)  # Increase contrast by 20%

        # Sharpening
        if sharpen:
            pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # Color enhancement for aurora
        color_enhancer = ImageEnhance.Color(pil_image)
        pil_image = color_enhancer.enhance(1.1)  # Slight color boost

        return np.array(pil_image)

    def process_single_image(self, image_path: str) -> ImageAnalysis:
        """Process a single image with all analysis functions"""
        # Load image
        image = self.load_image(image_path)
        filename = Path(image_path).name

        print(f"Processing {filename}...")

        # Analyze composition
        composition = self.analyze_composition(image)

        # Analyze RGB intensities
        rgb_stats = self.analyze_rgb_intensities(image)

        # Create color histogram
        hist_save_path = self.output_folder / "visualizations" / f"{Path(filename).stem}_histogram.png"
        self.create_color_histogram(image, str(hist_save_path))

        # Detect aurora intensity
        aurora_intensity = self.detect_aurora_intensity(image)

        # Get dominant colors
        dominant_colors = self.get_dominant_colors(image)

        # Clean image
        cleaned_image = self.clean_image(image)

        # Save cleaned image
        cleaned_path = self.output_folder / "cleaned" / f"cleaned_{filename}"
        cv2.imwrite(str(cleaned_path), cv2.cvtColor(cleaned_image, cv2.COLOR_RGB2BGR))

        # Create analysis object
        analysis = ImageAnalysis(
            filename=filename,
            dimensions=(image.shape[1], image.shape[0]),
            brightness_mean=rgb_stats['overall']['brightness_mean'],
            brightness_std=rgb_stats['overall']['brightness_std'],
            rgb_means=(rgb_stats['red']['mean'], rgb_stats['green']['mean'], rgb_stats['blue']['mean']),
            rgb_stds=(rgb_stats['red']['std'], rgb_stats['green']['std'], rgb_stats['blue']['std']),
            dominant_colors=dominant_colors,
            composition_score=composition['overall_composition_score'],
            aurora_intensity=aurora_intensity
        )

        # Save detailed analysis
        analysis_data = {
            'filename': filename,
            'dimensions': analysis.dimensions,
            'composition_analysis': composition,
            'rgb_statistics': rgb_stats,
            'aurora_intensity': aurora_intensity,
            'dominant_colors': dominant_colors
        }

        analysis_path = self.output_folder / "analysis" / f"{Path(filename).stem}_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        return analysis

    def process_all_images(self, image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']) -> List[
        ImageAnalysis]:
        """Process all images in the input folder"""
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(str(self.input_folder / f"*{ext}")))
            image_files.extend(glob.glob(str(self.input_folder / f"*{ext.upper()}")))

        image_files.sort()  # Sort for consistent ordering

        if not image_files:
            raise ValueError(f"No images found in {self.input_folder}")

        print(f"Found {len(image_files)} images to process")

        analyses = []
        for image_path in image_files:
            try:
                analysis = self.process_single_image(image_path)
                analyses.append(analysis)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        # Save summary analysis
        self.save_summary_analysis(analyses)

        return analyses

    def save_summary_analysis(self, analyses: List[ImageAnalysis]):
        """Save summary statistics for all processed images"""
        if not analyses:
            return

        summary = {
            'total_images': len(analyses),
            'average_brightness': np.mean([a.brightness_mean for a in analyses]),
            'average_aurora_intensity': np.mean([a.aurora_intensity for a in analyses]),
            'best_composition_image': max(analyses, key=lambda x: x.composition_score).filename,
            'highest_aurora_intensity_image': max(analyses, key=lambda x: x.aurora_intensity).filename,
            'rgb_statistics': {
                'red_mean': np.mean([a.rgb_means[0] for a in analyses]),
                'green_mean': np.mean([a.rgb_means[1] for a in analyses]),
                'blue_mean': np.mean([a.rgb_means[2] for a in analyses]),
            }
        }

        summary_path = self.output_folder / "analysis" / "analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Summary analysis saved to {summary_path}")

    def create_timelapse(self, fps: int = 24, output_name: str = "northern_lights_timelapse.mp4",
                         use_cleaned: bool = True) -> str:
        """Create a timelapse video from processed images"""
        # Choose image source
        if use_cleaned:
            image_folder = self.output_folder / "cleaned"
            pattern = "cleaned_*"
        else:
            image_folder = self.input_folder
            pattern = "*"

        # Get all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            image_files.extend(glob.glob(str(image_folder / f"{pattern}{ext}")))
            image_files.extend(glob.glob(str(image_folder / f"{pattern}{ext.upper()}")))

        image_files.sort()

        if not image_files:
            raise ValueError(f"No images found in {image_folder}")

        print(f"Creating timelapse from {len(image_files)} images...")

        # Read first image to get dimensions
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = self.output_folder / output_name
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Add each image to video
        for i, image_file in enumerate(image_files):
            img = cv2.imread(image_file)
            if img is not None:
                # Resize if necessary
                if img.shape[:2] != (height, width):
                    img = cv2.resize(img, (width, height))
                video_writer.write(img)

            if i % 10 == 0:
                print(f"Processed {i}/{len(image_files)} frames")

        video_writer.release()
        print(f"Timelapse saved as {output_path}")

        return str(output_path)

    def create_analysis_visualization(self):
        """Create comprehensive visualization of analysis results"""
        # Load all analysis files
        analysis_files = glob.glob(str(self.output_folder / "analysis" / "*_analysis.json"))
        if not analysis_files:
            print("No analysis files found for visualization")
            return

        analysis_files.sort()

        analyses_data = []
        for file in analysis_files:
            with open(file, 'r') as f:
                analyses_data.append(json.load(f))

        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Northern Lights Image Analysis Summary', fontsize=16)

        # 1. Aurora intensity over time
        aurora_intensities = [data['aurora_intensity'] for data in analyses_data]
        axes[0, 0].plot(aurora_intensities, marker='o')
        axes[0, 0].set_title('Aurora Intensity Over Time')
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Aurora Intensity')

        # 2. Brightness distribution
        brightness_means = [data['rgb_statistics']['overall']['brightness_mean'] for data in analyses_data]
        axes[0, 1].hist(brightness_means, bins=20, alpha=0.7)
        axes[0, 1].set_title('Brightness Distribution')
        axes[0, 1].set_xlabel('Mean Brightness')
        axes[0, 1].set_ylabel('Frequency')

        # 3. RGB channel comparison
        red_means = [data['rgb_statistics']['red']['mean'] for data in analyses_data]
        green_means = [data['rgb_statistics']['green']['mean'] for data in analyses_data]
        blue_means = [data['rgb_statistics']['blue']['mean'] for data in analyses_data]

        x = range(len(analyses_data))
        axes[0, 2].plot(x, red_means, 'r-', label='Red', alpha=0.7)
        axes[0, 2].plot(x, green_means, 'g-', label='Green', alpha=0.7)
        axes[0, 2].plot(x, blue_means, 'b-', label='Blue', alpha=0.7)
        axes[0, 2].set_title('RGB Channel Means Over Time')
        axes[0, 2].set_xlabel('Image Index')
        axes[0, 2].set_ylabel('Mean Intensity')
        axes[0, 2].legend()

        # 4. Composition scores
        composition_scores = [data['composition_analysis']['overall_composition_score'] for data in analyses_data]
        axes[1, 0].bar(range(len(composition_scores)), composition_scores)
        axes[1, 0].set_title('Composition Scores')
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Composition Score')

        # 5. Aurora vs Brightness correlation
        axes[1, 1].scatter(brightness_means, aurora_intensities, alpha=0.6)
        axes[1, 1].set_title('Aurora Intensity vs Brightness')
        axes[1, 1].set_xlabel('Mean Brightness')
        axes[1, 1].set_ylabel('Aurora Intensity')

        # 6. Color space analysis (dominant colors)
        # Show average dominant colors as color patches
        all_dominant_colors = []
        for data in analyses_data:
            all_dominant_colors.extend(data['dominant_colors'])

        # Average the colors
        if all_dominant_colors:
            avg_colors = np.mean(all_dominant_colors, axis=0).reshape(1, 1, 3).astype(np.uint8)
            axes[1, 2].imshow(avg_colors)
            axes[1, 2].set_title('Average Dominant Color Palette')
            axes[1, 2].axis('off')

        plt.tight_layout()

        viz_path = self.output_folder / "visualizations" / "comprehensive_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Comprehensive analysis visualization saved to {viz_path}")


# Example usage
def main():
    """Main function demonstrating usage"""
    # Initialize processor
    processor = NorthernLightsProcessor(
        input_folder="data",
        output_folder="northern_lights_processed"
    )

    # Process all images
    analyses = processor.process_all_images()

    # Create timelapse
    timelapse_path = processor.create_timelapse(fps=30, use_cleaned=True)

    # Create analysis visualization
    processor.create_analysis_visualization()

    print(f"Processing complete!")
    print(f"Processed {len(analyses)} images")
    print(f"Timelapse saved to: {timelapse_path}")
    print(f"All outputs saved to: {processor.output_folder}")


if __name__ == "__main__":
    main()
