import os
import json
import logging
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
HISTORY_FILE = "history.json"

# Configure OpenAI client for NVIDIA API
client = None
if OPENAI_API_KEY:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=OPENAI_API_KEY
    )

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def fetch_trending_news():
    """Fetches the latest news about Football/FIFA World Cup from Google News RSS."""
    url = "https://news.google.com/rss/search?q=FIFA+World+Cup+Football+when:1d&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll('item')
        
        news_list = []
        for item in items:
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else ""
            pub_date = item.pubDate.text if item.pubDate else ""
            description = item.description.text if item.description else ""
            news_list.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "description": description
            })
        return news_list
    except Exception as e:
        logging.error(f"Error fetching news: {e}")
        return []

def extract_og_image(url):
    """Attempts to extract the og:image from the article URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        og_img = soup.find('meta', property='og:image')
        if og_img and og_img.get('content'):
            return og_img['content']
    except Exception as e:
        logging.warning(f"Could not extract og:image for {url}: {e}")
    return None

def generate_facebook_post(news_item):
    """Uses OpenAI to generate a highly engaging Facebook post in American English."""
    if not client:
        logging.error("OpenAI client is not initialized (missing API key).")
        return None
        
    prompt = f"""
    You are an expert social media manager for a huge football (soccer) fan page in the United States.
    Your goal is to write a short, simple, and highly engaging Facebook text post about the following news.
    
    News Title: {news_item['title']}
    News Snippet: {news_item['description']}
    
    Requirements:
    - KEEP IT VERY SHORT AND SIMPLE. People should be able to read it instantly and react. Do not write a long post.
    - Target Audience: United States (American Audience).
    - Primary Goal: Maximize engagement by asking a direct, thought-provoking question to the audience.
    - Make the language easy, attractive, and click-worthy.
    - Include relevant emojis to boost engagement.
    - MUST include a strong Call-to-Action (CTA) or question at the end to inspire Comments, Likes, or Shares.
    - The output must ONLY be the text of the Facebook post itself, ready to be published. Do not include any meta-text, quotes, or notes.
    """
    
    try:
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[
                {"role": "system", "content": "You are a professional social media manager."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating post with OpenAI: {e}")
        return None

def send_to_discord(post_text, image_url, news_url):
    """Sends the generated post and image to Discord via Webhook."""
    if not DISCORD_WEBHOOK_URL:
        logging.error("Discord Webhook URL is missing.")
        return False
        
    payload = {
        "content": f"{post_text}\n\n🔗 **Read more:** {news_url}"
    }
    
    if image_url:
        payload["embeds"] = [
            {
                "image": {
                    "url": image_url
                }
            }
        ]
        
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Successfully sent message to Discord.")
        return True
    except Exception as e:
        logging.error(f"Error sending to Discord: {e}")
        return False

def job():
    logging.info("Starting the news fetch and post generation job...")
    
    history = load_history()
    news_items = fetch_trending_news()
    
    if not news_items:
        logging.info("No news found.")
        return
        
    # Find the first news item that hasn't been posted yet
    new_item = None
    for item in news_items:
        if item['link'] not in history:
            new_item = item
            break
            
    if not new_item:
        logging.info("No new articles found. All recent articles have already been posted.")
        return
        
    logging.info(f"Found new article: {new_item['title']}")
    
    # Generate Post
    logging.info("Generating Facebook post via LLM...")
    post_text = generate_facebook_post(new_item)
    
    if not post_text:
        logging.error("Failed to generate post text.")
        return
        
    # Extract Image
    logging.info("Extracting image from article...")
    image_url = extract_og_image(new_item['link'])
    
    # Send to Discord
    logging.info("Sending to Discord...")
    success = send_to_discord(post_text, image_url, new_item['link'])
    
    if success:
        # Save to history to avoid duplicate
        history.append(new_item['link'])
        # Keep history file from growing infinitely, store last 500
        if len(history) > 500:
            history = history[-500:]
        save_history(history)
        logging.info("Job completed successfully.")
    else:
        logging.error("Job failed at Discord sending stage.")

def main():
    logging.info("Football News Bot initialized.")
    # Run once
    job()
    logging.info("Run completed.")

if __name__ == "__main__":
    main()
