import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to preprocess the image, create a mask, and extract the main content
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Perform morphological opening
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours to get the bounding box of the main content
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image, (x, y, w, h), contours
    else:
        return image, None, contours  # Return the original if no contours found

# Function to calculate angle between two lines given the endpoints
def calculate_angle(p1, p2, p3):
    # Calculate the angle between line p1p2 and line p2p3
    line1 = (p1[0] - p2[0], p1[1] - p2[1])
    line2 = (p3[0] - p2[0], p3[1] - p2[1])
    angle = np.degrees(np.arctan2(line2[1], line2[0]) - np.arctan2(line1[1], line1[0]))
    if angle < 0:
        angle += 360
    return angle

# Function to compare images, draw lines on outer shell, and segment inner differences
def compare_images(image1_path, image2_path, output_folder, area_threshold=10, diff_threshold=30):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    # Ensure the images are loaded
    if image1 is None or image2 is None:
        print("Error: One or both image paths are incorrect.")
        return
    
    # Preprocess images to extract main content
    image1_cropped, image1_outer_box, contours1 = preprocess_image(image1)
    image2_cropped, image2_outer_box, contours2 = preprocess_image(image2)
    
    # Resize second image to match the first
    image2_resized = cv2.resize(image2_cropped, (image1_cropped.shape[1], image1_cropped.shape[0]))
    
    # Calculate angle between the sides of the bounding box (if a rectangle)
    if image1_outer_box and image2_outer_box:
        angle1 = calculate_angle((image1_outer_box[0], image1_outer_box[1]), 
                                 (image1_outer_box[0] + image1_outer_box[2], image1_outer_box[1]),
                                 (image1_outer_box[0], image1_outer_box[1] + image1_outer_box[3]))
        
        angle2 = calculate_angle((image2_outer_box[0], image2_outer_box[1]), 
                                 (image2_outer_box[0] + image2_outer_box[2], image2_outer_box[1]),
                                 (image2_outer_box[0], image2_outer_box[1] + image2_outer_box[3]))
        
        print(f"Angle of Image 1 outer shell: {angle1:.2f} degrees")
        print(f"Angle of Image 2 outer shell: {angle2:.2f} degrees")
    
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1_cropped, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold for major differences
    _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of the differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes on the inner differences (filter based on area)
    image1_boxes = image1_cropped.copy()
    image2_boxes = image2_resized.copy()
    
    for contour in contours:
        if cv2.contourArea(contour) > area_threshold:  # Filter based on area threshold
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image1_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(image2_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Create an overlapping segmentation highlighting differences in white
    segmented_image = image1_cropped.copy()
    segmented_image[thresh > 0] = [255, 255, 255]  # Highlight differences in white
    cv2.imwrite(r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\airlab\examples\Input\image1.jpg",image1_cropped)
    cv2.imwrite(r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\airlab\examples\Input\image2.jpg",image2_resized)
    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image1_cropped, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Image 1 Cropped")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(image2_resized, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Image 2 Cropped")
    axs[1].axis("off")

    axs[2].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
    axs[2].set_title("Segmented Image with Inner Differences")
    axs[2].axis("off")

    plt.show()

    
    # Save output images
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, "C:/Users/Kartikey.Tiwari/Downloads/Crimp_Cross_Section/segmented_output/diff1.jpg"), image1_boxes)
    cv2.imwrite(os.path.join(output_folder, "C:/Users/Kartikey.Tiwari/Downloads/Crimp_Cross_Section/segmented_output/diff2.jpg"), image2_boxes)
    cv2.imwrite(os.path.join(output_folder, "C:/Users/Kartikey.Tiwari/Downloads/Crimp_Cross_Section/segmented_output/segmented_image.jpg"), segmented_image)
    print(f"Results saved in {output_folder}")

# if __name__ == "__main__":
#     compare_images(r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\data\CCSES - 23-JAN-25\OK,NG SAMPLE_NEXT lot\Terminal sheet cracked\Case 3\CS14001930_FLRY B 0050_CH 0090_OK.JPG", r"C:\Users\Kartikey.Tiwari\Downloads\Crimp_Cross_Section\data\CCSES - 23-JAN-25\OK,NG SAMPLE_NEXT lot\Terminal sheet cracked\Case 3\CS14001930_FLRY B 0050_CH 0085_NG.JPG", "output_folder_path")

if __name__ == "__main__":
    compare_images("C:/Users/Kartikey.Tiwari/Downloads/Crimp_Cross_Section/airlab/examples/Input/airok.jpg", "C:/Users/Kartikey.Tiwari/Downloads/Crimp_Cross_Section/airlab/examples/Input/airng.jpg", "output_folder_path")
