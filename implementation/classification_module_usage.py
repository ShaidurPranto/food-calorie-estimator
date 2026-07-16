import os
from classification_module import FoodClassifier


def main():
    """
    Main function to demonstrate food classification on a sample image.
    """
    # Configuration
    IMAGE_PATH = "random/orange3.jpg"

    # Initialize classifier
    print("Initializing Food Classifier...")
    classifier = FoodClassifier()
    print("✓ Classifier initialized successfully")

    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    # Classify image by path
    print(f"\nClassifying image: {IMAGE_PATH}")
    result = classifier.classify_by_path(IMAGE_PATH)

    # Display results
    print(f"Classification Results:")
    print(f"  - Image Path: {result['image_path']}")
    print(f"  - Predicted Class: {result['class_name']} (Index {result['class_index']})")
    print(f"  - Confidence: {result['confidence']:.4f}")


if __name__ == "__main__":
    main()
