"""
Enhanced web scraper to collect REAL Canadian mining data.
This script actively scrapes from available public sources.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import time
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealDataScraper:
    """Scraper for real Canadian mining data from public sources."""
    
    def __init__(self, delay=2.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.all_records = []
    
    def scrape_open_canada_mining(self):
        """
        Scrape mining data from Open Canada portal.
        Uses the CKAN API to search for mining datasets.
        """
        logger.info("🔍 Searching Open Canada portal for mining data...")
        
        try:
            # Open Canada CKAN API
            api_url = "https://open.canada.ca/data/api/3/action/package_search"
            
            # Search for mining-related datasets
            params = {
                "q": "mining mines mineral exploration",
                "rows": 20,
                "sort": "metadata_modified desc"
            }
            
            response = self.session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data['success']:
                datasets = data['result']['results']
                logger.info(f"✅ Found {len(datasets)} mining-related datasets")
                
                # Look for downloadable CSV/JSON resources
                for dataset in datasets[:5]:  # Check first 5 datasets
                    title = dataset.get('title', 'Unknown')
                    logger.info(f"   📦 {title}")
                    
                    resources = dataset.get('resources', [])
                    for resource in resources:
                        format_type = resource.get('format', '').lower()
                        if format_type in ['csv', 'json', 'xlsx']:
                            url = resource.get('url', '')
                            logger.info(f"      📄 Found {format_type.upper()} file: {resource.get('name', 'Unnamed')}")
                            
                            # Try to download and parse
                            if format_type == 'csv':
                                df = self._download_csv(url, f"open_canada_{dataset.get('name', 'data')}")
                                if df is not None and len(df) > 0:
                                    df['source'] = 'Open Canada - ' + title
                                    self.all_records.append(df)
                                    logger.info(f"      ✅ Collected {len(df)} records")
                                    time.sleep(self.delay)
                                    break  # Got data from this dataset, move to next
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error scraping Open Canada: {e}")
            return False
    
    def scrape_nrcan_major_projects(self):
        """
        Scrape NRCan Major Projects data.
        NRCan maintains a list of major resource projects.
        """
        logger.info("🔍 Scraping NRCan major projects...")
        
        try:
            # NRCan often publishes data files directly or through Open Canada
            # Try to find their major projects inventory
            
            # Option 1: Check if there's a direct data file
            possible_urls = [
                "https://www.nrcan.gc.ca/sites/nrcan/files/mineralsmetals/files/pdf/mmi-emi/major-projects-inventory-eng.csv",
                "https://open.canada.ca/data/en/dataset/1cddadb6-f34a-4c2e-92ae-f8cfb2cdef66",  # Major projects dataset
            ]
            
            # Try the Open Canada dataset for NRCan major projects
            nrcan_api_url = "https://open.canada.ca/data/api/3/action/package_show"
            params = {"id": "major-projects"}
            
            response = self.session.get(nrcan_api_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    resources = data['result'].get('resources', [])
                    for resource in resources:
                        if resource.get('format', '').lower() in ['csv', 'json']:
                            url = resource.get('url')
                            logger.info(f"   📄 Found NRCan data: {resource.get('name')}")
                            
                            df = self._download_csv(url, "nrcan_major_projects")
                            if df is not None and len(df) > 0:
                                # Filter for mining projects
                                if 'sector' in df.columns:
                                    df = df[df['sector'].str.contains('mining|mineral', case=False, na=False)]
                                df['source'] = 'NRCan Major Projects'
                                self.all_records.append(df)
                                logger.info(f"   ✅ Collected {len(df)} mining project records")
                                return True
            
            logger.warning("⚠️  Could not access NRCan direct data, trying alternative...")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error scraping NRCan: {e}")
            return False
    
    def scrape_bc_open_data(self):
        """
        Scrape BC Open Data for mining information.
        BC has excellent open data for mines.
        """
        logger.info("🔍 Scraping BC Open Data...")
        
        try:
            # BC Data Catalogue CKAN API
            api_url = "https://catalogue.data.gov.bc.ca/api/3/action/package_show"
            
            # Major mines dataset
            params = {"id": "major-mines"}
            
            response = self.session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data['success']:
                resources = data['result']['resources']
                logger.info(f"✅ Found BC Major Mines dataset with {len(resources)} resources")
                
                for resource in resources:
                    format_type = resource.get('format', '').lower()
                    if format_type in ['csv', 'json']:
                        url = resource.get('url')
                        name = resource.get('name', 'BC data')
                        logger.info(f"   📄 Downloading: {name}")
                        
                        df = self._download_csv(url, "bc_major_mines")
                        if df is not None and len(df) > 0:
                            df['source'] = 'BC Open Data - Major Mines'
                            self.all_records.append(df)
                            logger.info(f"   ✅ Collected {len(df)} BC mine records")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error scraping BC data: {e}")
            return False
    
    def scrape_mining_association_data(self):
        """
        Try to get data from mining associations that publish open data.
        """
        logger.info("🔍 Checking mining association data sources...")
        
        try:
            # Mining Association of Canada sometimes publishes data
            # Fraser Institute mining survey data might be available
            
            # This is a placeholder - these would need specific implementations
            logger.info("   ℹ️  Mining association data typically requires manual download")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error checking association data: {e}")
            return False
    
    def scrape_provincial_data(self):
        """
        Try to scrape from other provincial open data portals.
        """
        logger.info("🔍 Checking other provincial sources...")
        
        provincial_apis = {
            'Ontario': 'https://data.ontario.ca/api/3/action/package_search',
            'Quebec': 'https://www.donneesquebec.ca/recherche/api/3/action/package_search',
            'Alberta': 'https://open.alberta.ca/api/3/action/package_search',
        }
        
        for province, api_url in provincial_apis.items():
            try:
                logger.info(f"   🔍 Searching {province}...")
                params = {"q": "mining mineral mines", "rows": 5}
                
                response = self.session.get(api_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        datasets = data['result'].get('results', [])
                        logger.info(f"      Found {len(datasets)} potential datasets")
                        
                        for dataset in datasets[:2]:  # Check first 2
                            resources = dataset.get('resources', [])
                            for resource in resources:
                                if resource.get('format', '').lower() == 'csv':
                                    url = resource.get('url')
                                    df = self._download_csv(url, f"{province.lower()}_mining")
                                    if df is not None and len(df) > 0:
                                        df['source'] = f'{province} Open Data'
                                        self.all_records.append(df)
                                        logger.info(f"      ✅ Collected {len(df)} records from {province}")
                                        break
                
                time.sleep(self.delay)
                
            except Exception as e:
                logger.warning(f"      ⚠️  Could not access {province}: {e}")
                continue
        
        return True
    
    def _download_csv(self, url, filename_prefix):
        """Download and parse a CSV file."""
        try:
            logger.info(f"      ⬇️  Downloading from {url[:80]}...")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to parse as CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            logger.info(f"      📊 Parsed {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.warning(f"      ⚠️  Could not download/parse CSV: {e}")
            return None
    
    def collect_all_data(self, output_path: Path, target_records=100):
        """
        Collect data from all available sources.
        """
        logger.info("="*80)
        logger.info("🚀 STARTING REAL DATA COLLECTION")
        logger.info("="*80)
        
        self.all_records = []
        
        # Try each source
        sources = [
            ("BC Open Data", self.scrape_bc_open_data),
            ("NRCan Major Projects", self.scrape_nrcan_major_projects),
            ("Open Canada Portal", self.scrape_open_canada_mining),
            ("Provincial Sources", self.scrape_provincial_data),
        ]
        
        for source_name, scrape_func in sources:
            logger.info(f"\n{'='*80}")
            logger.info(f"📍 Source: {source_name}")
            logger.info(f"{'='*80}")
            
            try:
                scrape_func()
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"❌ Error with {source_name}: {e}")
                continue
            
            # Check if we have enough records
            total_records = sum(len(df) for df in self.all_records)
            if total_records >= target_records:
                logger.info(f"\n✅ Target reached! Collected {total_records} records")
                break
        
        # Combine all data
        if self.all_records:
            combined_df = pd.concat(self.all_records, ignore_index=True)
            
            # Save raw data
            output_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"real_mining_data_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"\n{'='*80}")
            logger.info(f"✅ SUCCESS! Real data collection complete")
            logger.info(f"{'='*80}")
            logger.info(f"📊 Total records collected: {len(combined_df)}")
            logger.info(f"📁 Saved to: {output_file}")
            logger.info(f"📋 Columns: {list(combined_df.columns)}")
            logger.info(f"🗂️  Data sources: {combined_df['source'].unique().tolist()}")
            
            # Show preview
            logger.info(f"\n{'='*80}")
            logger.info("📋 DATA PREVIEW (first 5 rows):")
            logger.info(f"{'='*80}")
            print(combined_df.head())
            
            return combined_df
        else:
            logger.warning("\n⚠️  No data was collected from any source")
            logger.info("\n💡 TIP: Try manual download from:")
            logger.info("   - https://catalogue.data.gov.bc.ca/dataset/major-mines")
            logger.info("   - https://open.canada.ca/data/en/dataset/major-projects")
            return pd.DataFrame()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏔️  REAL CANADIAN MINING DATA SCRAPER")
    print("="*80)
    print("\n⚠️  This will attempt to scrape REAL data from public Canadian sources")
    print("    Please be patient, this may take a few minutes...\n")
    
    # Set up paths
    output_path = Path(__file__).parent.parent.parent / "data" / "raw"
    
    # Create scraper
    scraper = RealDataScraper(delay=2.0)
    
    # Collect data
    data = scraper.collect_all_data(output_path, target_records=100)
    
    print("\n" + "="*80)
    if len(data) > 0:
        print(f"✅ SUCCESS! Collected {len(data)} real mining records")
        print(f"📁 Check: data/raw/ directory")
    else:
        print("⚠️  No data collected. See suggestions above.")
    print("="*80 + "\n")
