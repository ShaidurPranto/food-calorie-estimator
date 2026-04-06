from filterization_module import FoodFilter

def main():
    print("Initializing FoodFilter...")
    food_filter = FoodFilter()
    print("FoodFilter initialized properly.")

    image_path = "000002.jpg"
    
    print(f"Checking image: {image_path}")
    try:
        is_food = food_filter.is_food(image_path)
        print(f"Result -> Is the image food? {is_food}")
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
