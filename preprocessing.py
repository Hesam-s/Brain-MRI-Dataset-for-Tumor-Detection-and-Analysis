import cv2
import gc
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#=================================================================================================
#                                 Noise Removal & Enhancement
#=================================================================================================
def apply_gaussian_denoising(image):
    # Keep the 3x3 Gaussian Blur as requested
    return cv2.GaussianBlur(image, (3, 3), 0)

def get_denoised_stack(X_raw):
    print("✨ Applying 3x3 Gaussian Noise Removal to Raw Stack...")
    all_processed = []

    for img in X_raw:
        clean_img = apply_gaussian_denoising(img)
        
        if len(clean_img.shape) == 2:
            clean_img = clean_img[:, :, np.newaxis]
            
        all_processed.append(clean_img)

    # FIX: Change this line to use dtype=object
    X_denoised = np.array(all_processed, dtype=object) 
    
    print(f"✅ Noise Removal Complete! Shape: {X_denoised.shape}")
    return X_denoised

#=================================================================================================
#                                     Data Resizing
#=================================================================================================
def resize_denoised_stack(denoised_array, target_size=(224, 224)):
    resized_list = []
    print(f"📏 Resizing images to {target_size} using High-Quality Lanczos...")

    for img in denoised_array:
        # INTER_LANCZOS4 is preferred in medical papers for preserving detail
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
        resized_list.append(resized_img)

    X_final = np.array(resized_list)
    return X_final

#=================================================================================================
#                                  Improved Normalization
#=================================================================================================
def normalize_stack(X_final):
    # 1. Properly scale the data to exactly [0, 1]
    # This handles raw MRI values that exceed 255
    x_min = X_final.min()
    x_max = X_final.max()
    X_normalized = (X_final.astype('float32') - x_min) / (x_max - x_min)

    print("✨ Performing Corrected Data Normalization...")
    print(f"✅ Normalization Complete!")
    print(f"Final Data Range: {X_normalized.min()} to {X_normalized.max()}")
    print(f"Final Tensor Shape: {X_normalized.shape}")
    
    return X_normalized

#=================================================================================================
#                           RGB Channel Conversion (ViT Compatibility)
#=================================================================================================
def convert_to_rgb(X_normalized):
    # If already 3 channels (Batch, H, W, 3), don't repeat
    if X_normalized.shape[-1] == 1:
        X_rgb = np.repeat(X_normalized, 3, axis=-1)
    else:
        X_rgb = X_normalized

    # contrast recovery
    for i in range(len(X_rgb)):
        img_min = X_rgb[i].min()
        img_max = X_rgb[i].max()
        if img_max - img_min > 0:
            X_rgb[i] = (X_rgb[i] - img_min) / (img_max - img_min)

    print("🎨 Contrast Recovery Complete.")
    return X_rgb

#=================================================================================================
#                                     Data Augmentation
#=================================================================================================
def get_augmentation_pipeline(X_train, y_train, batch_size=32):
    # 1. Force cast to float32 immediately to prevent underflow (black images)
    # This uses ~2.97 GB for your 4946 images
    if X_train.dtype != 'float32':
        X_train = X_train.astype('float32')
        gc.collect()

    # Define parameters based on your instructions
    datagen = ImageDataGenerator(
        rotation_range=30,      # 15 to 30 degrees
        horizontal_flip=True,   # Mirrored views
        vertical_flip=True,
        zoom_range=0.2,         # Partial zooming
        fill_mode='nearest'
    )

    n_images = X_train.shape[0]
    chunk_size = 300 # Requested chunk size
    print(f"🔄 Augmenting {n_images} images in-place using {chunk_size}-image chunks...")

    # 2. Process in chunks to keep temporary "math RAM" very low
    for i in range(0, n_images, chunk_size):
        end = min(i + chunk_size, n_images)
        
        # Pull a temporary copy of the chunk to feed the generator
        X_chunk_temp = X_train[i:end].copy()
        y_chunk_temp = y_train[i:end].copy()
        
        # 3. Generate augmented versions for ONLY this chunk
        chunk_gen = datagen.flow(X_chunk_temp, y_chunk_temp, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 seed=42)
        
        # Collect and overwrite immediately
        collected_X = []
        batches_processed = 0
        for x_b, _ in chunk_gen:
            collected_X.append(x_b)
            batches_processed += x_b.shape[0]
            if batches_processed >= (end - i):
                break
        
        # 4. OVERWRITE the original slice in X_train with the new augmented versions
        X_train[i:end] = np.concatenate(collected_X)
        
        # 5. THE CLEANUP: Completely wipe the temporary buffers for this chunk
        collected_X.clear()
        del collected_X, X_chunk_temp, y_chunk_temp, chunk_gen
        gc.collect() 
        
        if (i // chunk_size) % 3 == 0 or end == n_images:
            print(f"✅ Replaced through image {end}/{n_images}...")

    print(f"🏁 Done! Final X_train shape: {X_train.shape} (float32)")
    return X_train, y_train