import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm
import re
import chardet

load_dotenv()
API_KEY = os.getenv('GOOGLE_API_KEY')

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found")

genai.configure(api_key=API_KEY)

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

class OptimizedGameProcessor:
    def __init__(self, batch_size=10, max_workers=5, checkpoint_file="checkpoint.csv"):
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.checkpoint_file = checkpoint_file

    def create_batch_prompt(self, countries):
        """ create a single prompt for multiple countries """
        country_list = "\n".join([f"{i+1}. {country}" for i, country in enumerate(countries)])

        prompt = f"""Analyze the following countries and provide information in the exact JSON format below.
    
Countries to analyze:
{country_list}

For each country provide:
- region: the geographical region where the country is located

Response format (valid JSON only):
{{
    "countries" : [
        {{"region": "Europe"}},
        {{"region": "Asia"}}
    ]
}}

Provide exactly {len(countries)} countries in the same order as listed above."""
        
        return prompt
    
    def parse_batch_response(self, response_text, countries):
        """ Parse the JSON response and extract country data"""
        try:
            cleaned_response = response_text.strip()

            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                if "countries" in data and len(data["countries"]) == len(countries):
                    results = []
                    valid_regions = ["Africa", "Asia", "The Caribbean", "Central America", "Europe", "North America", "Oceania", "South America"]
                    
                    for i, country_data in enumerate(data["countries"]):
                        region = country_data.get("region", "Not specified")
                        if region not in valid_regions:
                            region = "Not Specified"

                        results.append({
                            "country": countries[i],
                            "region": region
                        })

                    return results
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")

        return [{
            "country": country,
            "region": "Not Specified"
        } for country in countries]
    
    def get_batch_response(self, countries, max_retries=3):
        """Get AI response for a batch of countries"""
        prompt = self.create_batch_prompt(countries)

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return self.parse_batch_response(response.text, countries)
            except Exception as e:
                print(f"Batch error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        print(f"Failed to process batch: {countries}")
        return [{
            "country": country,
            "region": "Not Specified"
        } for country in countries]
    
    def process_batch(self, batch):
        """Process a single batch of countries"""
        countries = batch["country"].tolist()
        result = self.get_batch_response(countries)

        result_df = pd.DataFrame(result)

        merged_df = batch.merge(result_df, on='country', how='left')
        return merged_df

    def save_checkpoint(self, processed_df, batch_num, total_batches):
        """Save progress to checkpoint file"""
        try:
            processed_df.to_csv(self.checkpoint_file, index=False, encoding='utf-8')
            print(f"Checkpoint saved: batch {batch_num}/{total_batches}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load existing checkpoint if available"""
        try:
            if os.path.exists(self.checkpoint_file):
                df = pd.read_csv(self.checkpoint_file)
                print(f"Loaded checkpoint with {len(df)} countries")
                return df
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
        return None
    
    def process_country_parallel(self, df):
        """Process countries in parallel batches"""
        checkpoint_df = self.load_checkpoint()
        
        if checkpoint_df is not None:
            processed_countries = set(checkpoint_df['country'].tolist())
            remaining_df = df[~df['country'].isin(processed_countries)]
            if len(remaining_df) == 0:
                print("All countries already processed!")
                return checkpoint_df
            print(f"Resuming: {len(remaining_df)} countries remaining")
            df = remaining_df
        else:
            checkpoint_df = pd.DataFrame()

        batches = [df[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        total_batches = len(batches)

        print(f"Processing {len(df)} countries in {total_batches} batches of {self.batch_size}")

        all_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self.process_batch, batch): (i, batch)
                for i, batch in enumerate(batches)
            }

            with tqdm(total=total_batches, desc="Processing batches") as pbar:
                for future in future_to_batch:
                    try:
                        result = future.result()
                        all_results.append(result)

                        pbar.update(1)

                        # Save checkpoint every 3 batches
                        if len(all_results) % 3 == 0:
                            current_results = pd.concat([checkpoint_df] + all_results, ignore_index=True)
                            self.save_checkpoint(current_results, len(all_results), total_batches)
                    
                    except Exception as e:
                        batch_num, batch = future_to_batch[future]
                        print(f"Batch {batch_num} failed: {e}")
        
        final_df = pd.concat([checkpoint_df] + all_results, ignore_index=True)

        # Clean up checkpoint file
        try:
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
        except:
            pass
        
        return final_df
    
def main():
    input_filename = "bi.csv"
    output_filename = "complete.csv"

    processor = OptimizedGameProcessor(
        batch_size=8,
        max_workers=3,
        checkpoint_file="country_processing_checkpoint.csv"
    )

    try:
        print(f"Loading data from {input_filename}")
        
        # Try to detect encoding first
        try:
            detected_encoding = detect_encoding(input_filename)
            print(f"Detected encoding: {detected_encoding}")
            df = pd.read_csv(input_filename, encoding=detected_encoding)
        except:
            # If chardet is not available or fails, try common encodings
            print("Trying different encodings...")
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(input_filename, encoding=encoding)
                    print(f"Successfully loaded with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read file with any common encoding")
        
        print(f"Loaded {len(df)} countries")
        print("Columns:", df.columns.tolist())
        print("Sample data:")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: file '{input_filename}' not found.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    start_time = time.time()
    print("Starting optimized processing...")

    enhanced_df = processor.process_country_parallel(df)

    end_time = time.time()
    processing_time = end_time - start_time

    try:
        print(f"Saving results to {output_filename}")
        enhanced_df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\nProcess completed successfully in {processing_time:.2f} seconds!")
        print(f"Processed {len(enhanced_df)} countries")
        print(f"Average time per country: {processing_time/len(enhanced_df):.2f} seconds")
        print("\nSample results:")
        print(enhanced_df.head())
        
        # Show summary statistics
        if 'region' in enhanced_df.columns:
            print(f"\nRegion distribution:")
            print(enhanced_df['region'].value_counts().head())
        
    except Exception as e:
        print(f"Failed to save results: {e}")

if __name__ == "__main__":
    main()