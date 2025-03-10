import os
import json
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


""" After this script run the cluster.py script to cluster the book covers descriptions over time."""

class EnhancedBookCoverAnalyzer:
    def __init__(self):
        print("Initializing models...")
        
        # Initialize BLIP for detailed image description
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        

    def get_blip_description(self, image):
        """Generate mostly detailed description using BLIP"""
        try:
            descriptions = []
            prompts = [
                "A romantic fantasy cover featuring", 
                "The main object on the image is",
                "The main visual elements are", # Added this prompt to get more diverse descriptions as example how bad it also can work with the wrong prompts, that's why it was not used for the clustering and the enhanced_analysis_results.json was used for the clustering.
            ]
            
            for prompt in prompts:
                # Process image with the prompt
                inputs = self.blip_processor(images=image, text=prompt, return_tensors="pt")
                
                out = self.blip_model.generate(
                    **inputs,
                    max_length=150,
                    num_beams=5,
                    do_sample=True,  # Enable sampling
                    temperature=0.7,
                    num_return_sequences=1
                )
                descriptions.append(self.blip_processor.decode(out[0], skip_special_tokens=True))
            
            return descriptions
        except Exception as e:
            print(f"Error in BLIP description generation: {str(e)}")
            return ["Error generating description"]

    def analyze_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Get BLIP descriptions
            descriptions = self.get_blip_description(image)
            
            # Get BLIP category analysis
            results = {
                "blip_descriptions": descriptions,
            }
            
            return results, image

        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            return None, None

    def visualize_results(self, image, results, save_path):
        n_categories = 1  # Only BLIP descriptions
        fig = plt.figure(figsize=(15, 5 + (n_categories * 2)))
        
        # Original image
        ax1 = plt.subplot2grid((n_categories + 1, 2), (0, 0))
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title('Original Book Cover')
        
        # BLIP Descriptions
        ax_blip = plt.subplot2grid((n_categories + 1, 2), (0, 1))
        ax_blip.axis('off')
        ax_blip.text(0, 0.8, "BLIP Descriptions:", fontweight='bold')
        for idx, desc in enumerate(results["blip_descriptions"], 1):
            ax_blip.text(0, 0.8 - (idx * 0.2), f"{idx}. {desc}", wrap=True)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def process_folder(self, folder_path):
        """Process all images in a folder"""
        results = {}
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(( '.jpg'))]
        
        print(f"\nFound {len(image_files)} images")
        
        viz_folder = os.path.join(folder_path, 'enhanced_visualizations')
        os.makedirs(viz_folder, exist_ok=True)
        
        for image_file in tqdm(image_files, desc="Analyzing book covers"):
            image_path = os.path.join(folder_path, image_file)
            image_results, image = self.analyze_image(image_path)
            
            if image_results and image:
                results[image_file] = image_results
                
                viz_path = os.path.join(
                    viz_folder, 
                    f'{os.path.splitext(image_file)[0]}_enhanced_analysis.jpg'
                )
                self.visualize_results(image, image_results, viz_path)

        output_file = os.path.join(folder_path, 'enhanced_analysis_results_new.json') # From the new run with three descriptions, but what was used for the clustering is 'enhanced_analysis_results.json, because the first two descriptions worked much better.
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Visualizations saved to: {viz_folder}")
        
        return results

if __name__ == "__main__":
    analyzer = EnhancedBookCoverAnalyzer()
    folder_path = "images"  
    results = analyzer.process_folder(folder_path)