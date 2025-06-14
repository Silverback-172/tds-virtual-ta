import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import requests
import json
import time
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from config.settings import settings

class DiscourseScraperFixed:
    def __init__(self):
        self.session = requests.Session()
        self.base_url = "https://discourse.onlinedegree.iitm.ac.in"
        self.category_id = 34  # TDS category
        # Make timezone-aware dates to fix parsing error
        self.start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
        self.end_date = datetime(2025, 4, 14, tzinfo=timezone.utc)
        
        # FRESH COOKIES FROM YOUR BROWSER
        self.cookies = {
            '_forum_session': 'XRkvbDZemeBiQxMu68r%2FD26J09oSJf%2Bsow9dCV9QzkQ6GganJin0bvqj5RJmY00GJnJL3MVQiiyWAJJWVO1hpTirCLxZiGk7dURRePvzNlxFu6BffWzG6CduzsWJU2CObSuyJNNynB5%2F6ncxnOx%2B7tKQhCWZ8FuYyPEaraDCds%2F6aDQj3%2FPyl3n942Oy42rtYn%2F0RajZyoLu%2Fol1m7IGRdnAoCENrpyb%2B5%2BG7AD5vy6ZOb%2Bt%2BuWNQ9PcU7TkL0jjBbAzt98Lw1JHGCgqnyCDDKkLYqqkGw%3D%3D--pCro2MLCftIQN2Ux--lyIvhY%2BFo1u4s0dFXSmfIQ%3D%3D',
            '_t': 'TqV9Rwbfjy73WBY4utRY2cbxdSjpwH5rvZxnaMRUkp7kx0H1hRkdpvgykIPzGi33IFaJgF685vmUkEFfamDlYkwvHiiL5d6wXwOsl1Hl6xB9IXjygKbb3TGgyI7LMorORBEBD922ZVwoGXy7t63OsqqMLX9LcgIzz%2BEYRMRQYjoDGHinsCb5b5alYg4ZrpGjrndUwFTTLz4gaK7EZYB1UuO8svk4otjXsdE%2F8KCl8iO78WrrAA27CDGRKmQ8KwunHVdGRmxP8d2CKBaFvx0hsk374Nr%2BsVJHce4muQ2Bfkp1SpNfmLK0lFJ64f0%3D--qQiu0dqTOSQpbxoH--tjqB0fGAaJZQaGQwZGHOSA%3D%3D'
        }
        
        # Headers to avoid compression issues - CRITICAL FIX
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:139.0) Gecko/20100101 Firefox/139.0',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'identity',  # CRITICAL: No compression
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        }
    
    def update_cookies_from_response(self, response):
        """Update cookies from response headers (TA RECOMMENDED)"""
        if 'Set-Cookie' in response.headers:
            for cookie in response.headers.get('Set-Cookie', '').split(','):
                if '_forum_session=' in cookie:
                    session_cookie = cookie.split(';')[0].split('=', 1)[1]
                    self.cookies['_forum_session'] = session_cookie
                    print(f"✅ Updated forum session cookie")
                elif '_t=' in cookie:
                    t_cookie = cookie.split(';')[0].split('=', 1)[1]
                    self.cookies['_t'] = t_cookie
                    print(f"✅ Updated _t cookie")
    
    def decode_response_content(self, response):
        """Simplified response decoding - no compression handling needed"""
        try:
            return response.json()
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            print(f"Response content preview: {response.text[:200]}")
            return None
        except Exception as e:
            print(f"❌ Response decoding error: {e}")
            return None
    
    def get_all_topics_with_pagination(self):
        """Get all topics using pagination with proper date handling"""
        print("🔍 Fetching all topics with pagination and date filtering...")
        
        all_topics = []
        page = 0
        
        while True:
            url = f"{self.base_url}/c/courses/tds-kb/{self.category_id}.json"
            params = {'page': page} if page > 0 else {}
            
            try:
                print(f"📄 Fetching page {page}...")
                response = self.session.get(
                    url, 
                    cookies=self.cookies, 
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
                
                print(f"📄 Page {page}: Status {response.status_code}")
                
                if response.status_code == 200:
                    data = self.decode_response_content(response)
                    if data is None:
                        break
                    
                    topics = data.get('topic_list', {}).get('topics', [])
                    
                    if not topics:
                        print(f"📄 No more topics found on page {page}")
                        break
                    
                    print(f"📄 Found {len(topics)} topics on page {page}")
                    
                    # Filter by date range with proper timezone handling
                    filtered_topics = []
                    for topic in topics:
                        created_at = topic.get('created_at', '')
                        if created_at:
                            try:
                                # Parse the datetime and make it timezone-aware
                                topic_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                
                                if self.start_date <= topic_date <= self.end_date:
                                    filtered_topics.append(topic)
                                    print(f"  ✅ Topic {topic.get('id')}: {topic.get('title', 'Unknown')[:50]}...")
                            except Exception as e:
                                print(f"  ⚠️ Date parsing error for topic {topic.get('id')}: {e}")
                    
                    all_topics.extend(filtered_topics)
                    self.update_cookies_from_response(response)
                    
                    print(f"✅ Page {page}: Found {len(filtered_topics)} relevant topics")
                    
                    if len(topics) < 30:
                        print("📄 Reached last page")
                        break
                    
                    page += 1
                    time.sleep(2)  # Rate limiting
                    
                elif response.status_code == 403:
                    print("❌ 403 Forbidden: Cookies expired or invalid")
                    break
                else:
                    print(f"❌ Failed to get topics: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"❌ Error fetching page {page}: {e}")
                break
        
        print(f"🎉 Total relevant topics found: {len(all_topics)}")
        return all_topics
    
    def scrape_individual_topic(self, topic_id, topic_title):
        """Scrape individual topic as separate JSON (TA REQUIREMENT)"""
        print(f"📖 Scraping topic {topic_id}: {topic_title[:50]}...")
        
        # Check if already exists to avoid re-scraping
        filename = f"discourse_topic_{topic_id}.json"
        filepath = os.path.join(settings.RAW_DATA_PATH, filename)
        
        if os.path.exists(filepath):
            print(f"⏭️ Skipping topic {topic_id} (already exists)")
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        url = f"{self.base_url}/t/{topic_id}.json"
        
        try:
            response = self.session.get(
                url,
                cookies=self.cookies,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                topic_data = self.decode_response_content(response)
                if topic_data is None:
                    return None
                
                self.update_cookies_from_response(response)
                
                # Process posts and clean HTML content
                posts = topic_data.get('post_stream', {}).get('posts', [])
                processed_posts = []
                
                for post in posts:
                    raw_content = post.get('raw', '')
                    cooked_content = post.get('cooked', '')
                    cleaned_text = self.clean_html_content(cooked_content)
                    
                    processed_post = {
                        'post_number': post.get('post_number', 1),
                        'username': post.get('username', 'unknown'),
                        'created_at': post.get('created_at', ''),
                        'updated_at': post.get('updated_at', ''),
                        'raw_content': raw_content,
                        'cooked_content': cooked_content,
                        'cleaned_text': cleaned_text,
                        'reply_count': post.get('reply_count', 0),
                        'like_count': post.get('actions_summary', [{}])[0].get('count', 0) if post.get('actions_summary') else 0,
                        'trust_level': post.get('trust_level', 0)
                    }
                    processed_posts.append(processed_post)
                
                processed_topic = {
                    'id': topic_data.get('id'),
                    'title': topic_data.get('title', ''),
                    'slug': topic_data.get('slug', ''),
                    'created_at': topic_data.get('created_at', ''),
                    'last_posted_at': topic_data.get('last_posted_at', ''),
                    'posts_count': topic_data.get('posts_count', 0),
                    'reply_count': topic_data.get('reply_count', 0),
                    'like_count': topic_data.get('like_count', 0),
                    'views': topic_data.get('views', 0),
                    'category_id': topic_data.get('category_id'),
                    'url': f"{self.base_url}/t/{topic_id}",
                    'posts': processed_posts,
                    'scraped_at': datetime.now().isoformat(),
                    'total_posts_scraped': len(processed_posts)
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(processed_topic, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Saved topic {topic_id} with {len(processed_posts)} posts")
                return processed_topic
                
            else:
                print(f"❌ Failed to scrape topic {topic_id}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error scraping topic {topic_id}: {e}")
            return None
    
    def clean_html_content(self, html_content):
        """Clean HTML content to extract readable text"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
            
            return cleaned_text
        except Exception as e:
            print(f"⚠️ HTML cleaning error: {e}")
            return html_content
    
    def scrape_all_discourse_data(self):
        """Complete discourse scraping workflow following TA method"""
        print("🚀 Starting comprehensive Discourse scraping (TA Method)...")
        
        os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
        
        # Get all relevant topics
        topics = self.get_all_topics_with_pagination()
        
        if not topics:
            print("❌ No topics found in the specified date range")
            return
        
        print(f"📊 Found {len(topics)} topics to scrape")
        
        # Scrape each topic individually
        scraped_topics = []
        failed_topics = []
        
        for i, topic in enumerate(topics):
            topic_id = topic.get('id')
            topic_title = topic.get('title', 'Unknown')
            
            print(f"\n📖 Processing topic {i+1}/{len(topics)}: {topic_id}")
            
            topic_data = self.scrape_individual_topic(topic_id, topic_title)
            if topic_data:
                scraped_topics.append(topic_data)
            else:
                failed_topics.append(topic_id)
            
            time.sleep(2)  # Rate limiting
            
            if (i + 1) % 10 == 0:
                print(f"📊 Progress: {i+1}/{len(topics)} topics processed")
        
        # Save comprehensive summary
        summary = {
            'scraping_metadata': {
                'total_topics_found': len(topics),
                'total_topics_scraped': len(scraped_topics),
                'failed_topics': failed_topics,
                'date_range': {
                    'start': self.start_date.isoformat(),
                    'end': self.end_date.isoformat()
                },
                'scraped_at': datetime.now().isoformat()
            },
            'topics_summary': [
                {
                    'id': topic['id'],
                    'title': topic['title'],
                    'posts_count': topic['posts_count'],
                    'url': topic['url'],
                    'created_at': topic['created_at']
                }
                for topic in scraped_topics
            ]
        }
        
        summary_path = os.path.join(settings.RAW_DATA_PATH, 'discourse_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n🎉 Discourse scraping completed!")
        print(f"📊 Successfully scraped: {len(scraped_topics)} topics")
        print(f"📊 Failed topics: {len(failed_topics)}")
        print(f"📁 Files saved in: {settings.RAW_DATA_PATH}")
        
        return scraped_topics

if __name__ == "__main__":
    print("🚀 Starting Final Discourse Scraper...")
    print("✅ Using fresh cookies from browser")
    print("🔧 No compression handling - direct JSON parsing")
    print("📅 Date range: Jan 1 - Apr 14, 2025")
    
    scraper = DiscourseScraperFixed()
    scraper.scrape_all_discourse_data()
