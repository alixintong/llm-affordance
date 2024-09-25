import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, Blip2ForImageTextRetrieval, AutoProcessor, FlavaModel

def preprocess_image(input_folder, output_folder, target_size=(224,224), padding_color=(255,255,255)):
    """
    Preprocesses images:
    1. Resizing them to a target size
    2. Padding them if it is necessary
    3. Saves the processed images in the output folder
    
    Input
        input_folder (str): Path to the folder containing original images
        output_folder (str): Path to the folder to save preprocessed images
        target_size (int, optional): Target size for the image (Default is 224)
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Add more formats if needed
            image_path = os.path.join(input_folder, filename)

            with Image.open(image_path) as img:
                
                # Convert to RGB if necessary
                if img.mode == 'RGBA':
                    img = Image.alpha_composite(Image.new("RGBA", img.size, padding_color), img)
                    img = img.convert("RGB")
                    
                img_aspect = img.width / img.height
                target_aspect = target_size[0] / target_size[1]

                # Resize image
                if img_aspect > target_aspect:
                    new_width = target_size[0]
                    new_height = int(target_size[0] / img_aspect)
                else:
                    new_height = target_size[1]
                    new_width = int(target_size[1] * img_aspect)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Pad image
                left_padding = (target_size[0] - new_width) // 2
                right_padding = target_size[0] - new_width - left_padding
                top_padding = (target_size[1] - new_height) // 2
                bottom_padding = target_size[1] - new_height - top_padding

                padded_img = Image.new("RGB", target_size, color=(255,255,255))
                padded_img.paste(img, (left_padding, top_padding))

                # Save the preprocessed image
                filename = filename.split('.')[0] + '.png'
                save_path = os.path.join(output_folder, filename)
                padded_img.save(save_path)

def expand2square(pil_img, background_color=(255,255,255)):
    '''
    1. Expanding to a square images
    2. Make background color standard
    '''
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def setup_model(model_name):
    '''
    Checking models
    '''
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if model_name == "blip2":
        model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g")
        preprocess = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
    elif model_name == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    elif model_name == "flava":
        model = FlavaModel.from_pretrained("facebook/flava-full")
        preprocess = AutoProcessor.from_pretrained("facebook/flava-full")
    else:
        raise ValueError("Model not implemented")

    tokenizer = None
    
    model.eval()
    model.to(device)
    return model, preprocess, tokenizer, device

def analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder, relationship):
    '''
    1. Reads dataframe
    2. Extracts text and images for specified model
    '''
    df = pd.read_csv(csv_path)
    all_results = []

    for index, item in tqdm(df.iterrows(), total=len(df)):
        image_paths = [os.path.join(img_folder, path) for path in [item[relationship+'_image'], item['non-afforded_image']]]
        images = [Image.open(path) for path in image_paths]

        inputs = preprocess(text=[item['condition'].strip()] * len(images), images=images, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            results = outputs.logits_per_image.softmax(dim=1)
        
        # Extract probabilities for the 'afforded' and 'non-afforded' images
        all_results.append({
            relationship: results[0][0].item(),
            'non_afforded': results[0][1].item(),
            'prompt_type': item['prompt_type'],
            'group_id': item['group_id']
        })

    return pd.DataFrame(all_results)

def format_results(df, model_name, dataset, relationship):
    '''
    Melting & reformatting the result

    1. Select id and prompt type to be the id, afforded and non-afforded as the variables
    2. 36 conditions, 72 separate afforded and non-afforded conditions

    '''
    melted_df = pd.melt(df, id_vars = ["group_id",'prompt_type'], value_vars=[relationship, 'non_afforded'])

    #Rename columns
    melted_df['relationships'] = melted_df['variable']
    melted_df = melted_df.rename(columns={'value': 'probability'}).drop(columns=['variable'])
    print(melted_df)

    #Formatting
    melted_df = melted_df[["relationships", 'prompt_type', "probability", "group_id"]]
    melted_df["model"] = model_name
    melted_df["dataset"] = dataset
    return melted_df

def results_summary(df):
    '''
    Produce summary for given df
    '''
    summary = df[["relationships", "probability"]].groupby(["relationships"]).mean()
    return summary

def ttest(df, relationship):
    '''
    Conduct Independent T-Test
    '''
    from scipy.stats import ttest_ind
    other_relationship = df[df["relationships"] == relationship]["probability"]
    non_afforded = df[df["relationships"] == "non_afforded"]["probability"]
    t, p_t = ttest_ind(other_relationship, non_afforded)

    return t, p_t

def anova(df):
    '''
    Perform Two-Factor ANOVA
    '''
    import statsmodels.api as sm 
    from statsmodels.formula.api import ols
  
    # Performing two-way ANOVA 
    ### 'probability ~ relationships + prompt_type + relationship:prompt_type'
    model = ols('probability ~ C(relationships) + C(prompt_type) + C(relationships):C(prompt_type)', data=df).fit()
    anova_result = sm.stats.anova_lm(model, typ=2) 

    return anova_result

def plot_results(df, save_path=None):
    '''
    Plot and save results
    '''
    sns.barplot(data=df, x="relationships", y="probability", hue = "prompt_type")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

