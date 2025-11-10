"""
Enhanced NRCan data scraper for Canadian mining data across all provinces.
Focuses on Natural Resources Canada (NRCan) data sources.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NRCanDataCollector:
    """Collector for NRCan and federal Canadian mining data."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.all_records = []
    
    def search_open_canada_nrcan(self, query, max_datasets=10):
        """
        Search Open Canada for NRCan mining datasets.
        """
        logger.info(f"🔍 Searching Open Canada for: '{query}'")
        
        try:
            api_url = "https://open.canada.ca/data/api/3/action/package_search"
            params = {
                "q": query,
                "rows": max_datasets,
                "fq": "organization:nrcan-rncan",
                "sort": "metadata_modified desc"
            }
            
            response = self.session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data['success'] and data['result']['results']:
                datasets = data['result']['results']
                logger.info(f"✅ Found {len(datasets)} NRCan datasets")
                return datasets
            else:
                logger.warning("No datasets found")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Open Canada: {e}")
            return []
    
    def download_dataset_resources(self, dataset, max_records=None):
        """
        Download CSV/JSON resources from a dataset.
        """
        dataset_title = dataset.get('title', 'Unknown')
        logger.info(f"\n📦 Dataset: {dataset_title}")
        
        collected_data = []
        
        for resource in dataset.get('resources', []):
            resource_format = resource.get('format', '').upper()
            resource_name = resource.get('name', 'Unknown')
            resource_url = resource.get('url', '')
            
            if resource_format in ['CSV', 'JSON', 'GEOJSON']:
                logger.info(f"   ⬇️  Downloading {resource_format}: {resource_name[:60]}...")
                
                try:
                    if resource_format == 'CSV':
                        df = pd.read_csv(resource_url, nrows=max_records, low_memory=False, encoding='utf-8', on_bad_lines='skip')
                    elif resource_format in ['JSON', 'GEOJSON']:
                        df = pd.read_json(resource_url)
                        if max_records:
                            df = df.head(max_records)
                    
                    if not df.empty:
                        df['source_dataset'] = dataset_title
                        df['source_resource'] = resource_name
                        collected_data.append(df)
                        logger.info(f"      ✅ Collected {len(df)} records, {len(df.columns)} columns")
                    
                except Exception as e:
                    logger.warning(f"      ⚠️  Could not load: {str(e)[:100]}")
                    continue
        
        return collected_data
    
    def collect_mining_project_data(self, target_records=500):
        """
        Collect mining project data from NRCan and Open Canada.
        """
        logger.info("\n" + "="*80)
        logger.info("🏔️  COLLECTING CANADA-WIDE MINING DATA FROM NRCan")
        logger.info("="*80)
        
        all_datasets = []
        
        # Search queries for different types of mining data
        search_queries = [
            "mining major projects canada",
            "mineral exploration canada",
            "mining permits canada",
            "mineral production canada",
            "mining operations canada",
        ]
        
        for query in search_queries:
            logger.info(f"\n{'─'*80}")
            logger.info(f"🔍 Search: '{query}'")
            logger.info(f"{'─'*80}")
            
            datasets = self.search_open_canada_nrcan(query, max_datasets=5)
            
            for dataset in datasets:
                # Download resources from this dataset
                data_frames = self.download_dataset_resources(dataset, max_records=200)
                all_datasets.extend(data_frames)
                
                # Check if we have enough records
                total_records = sum(len(df) for df in all_datasets)
                if total_records >= target_records:
                    logger.info(f"\n✅ Target reached! Collected {total_records} records")
                    break
            
            if all_datasets and sum(len(df) for df in all_datasets) >= target_records:
                break
            
            time.sleep(1)  # Be respectful
        
        return all_datasets
    
    def collect_provincial_open_data(self):
        """
        Collect data from provincial open data portals.
        """
        logger.info("\n" + "="*80)
        logger.info("🗺️  COLLECTING PROVINCIAL MINING DATA")
        logger.info("="*80)
        
        all_data = []
        
        # BC Open Data
        logger.info("\n📍 British Columbia")
        try:
            bc_url = "https://catalogue.data.gov.bc.ca/api/3/action/package_search"
            params = {"q": "mining mines mineral", "rows": 5}
            response = self.session.get(bc_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    logger.info(f"   Found {len(data['result']['results'])} BC datasets")
                    
                    for dataset in data['result']['results'][:3]:
                        for resource in dataset.get('resources', []):
                            if resource['format'].upper() == 'CSV':
                                try:
                                    df = pd.read_csv(resource['url'], nrows=100)
                                    df['province'] = 'British Columbia'
                                    df['source'] = 'BC Open Data'
                                    all_data.append(df)
                                    logger.info(f"   ✅ Collected {len(df)} BC records")
                                    break
                                except:
                                    continue
        except Exception as e:
            logger.warning(f"   ⚠️  BC data error: {e}")
        
        time.sleep(1)
        
        # Ontario Open Data
        logger.info("\n📍 Ontario")
        try:
            ont_url = "https://data.ontario.ca/api/3/action/package_search"
            params = {"q": "mining mineral", "rows": 5}
            response = self.session.get(ont_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    logger.info(f"   Found {len(data['result']['results'])} Ontario datasets")
                    
                    for dataset in data['result']['results'][:3]:
                        for resource in dataset.get('resources', []):
                            if resource['format'].upper() in ['CSV', 'JSON']:
                                try:
                                    if resource['format'].upper() == 'CSV':
                                        df = pd.read_csv(resource['url'], nrows=100)
                                    else:
                                        df = pd.read_json(resource['url'])
                                        df = df.head(100)
                                    
                                    df['province'] = 'Ontario'
                                    df['source'] = 'Ontario Open Data'
                                    all_data.append(df)
                                    logger.info(f"   ✅ Collected {len(df)} Ontario records")
                                    break
                                except:
                                    continue
        except Exception as e:
            logger.warning(f"   ⚠️  Ontario data error: {e}")
        
        time.sleep(1)
        
        # Saskatchewan
        logger.info("\n📍 Saskatchewan")
        try:
            sask_url = "https://open.canada.ca/data/api/3/action/package_search"
            params = {"q": "saskatchewan mining", "rows": 5}
            response = self.session.get(sask_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if data['success'] and data['result']['results']:
                    logger.info(f"   Found {len(data['result']['results'])} Saskatchewan datasets")
                    
                    for dataset in data['result']['results'][:2]:
                        for resource in dataset.get('resources', []):
                            if resource['format'].upper() == 'CSV':
                                try:
                                    df = pd.read_csv(resource['url'], nrows=50)
                                    df['province'] = 'Saskatchewan'
                                    df['source'] = 'Open Canada'
                                    all_data.append(df)
                                    logger.info(f"   ✅ Collected {len(df)} Saskatchewan records")
                                    break
                                except:
                                    continue
        except Exception as e:
            logger.warning(f"   ⚠️  Saskatchewan data error: {e}")
        
        return all_data


def main():
    """Main execution function."""
    
    output_path = Path(__file__).parent.parent.parent / "data" / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    collector = NRCanDataCollector()
    
    try:
        # Collect NRCan data
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING CANADA-WIDE DATA COLLECTION")
        logger.info("="*80)
        
        nrcan_data = collector.collect_mining_project_data(target_records=500)
        
        # Collect provincial data
        provincial_data = collector.collect_provincial_open_data()
        
        # Combine all data
        all_data = nrcan_data + provincial_data
        
        if all_data:
            # Combine into one DataFrame
            logger.info("\n" + "="*80)
            logger.info("📊 COMBINING ALL COLLECTED DATA")
            logger.info("="*80)
            
            combined_df = pd.concat(all_data, ignore_index=True, sort=False)
            
            # Save raw data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"canada_wide_mining_data_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"\n✅ SUCCESS!")
            logger.info("="*80)
            logger.info(f"📁 Saved to: {output_file}")
            logger.info(f"📊 Total records: {len(combined_df)}")
            logger.info(f"📋 Total columns: {len(combined_df.columns)}")
            
            # Show column names
            logger.info(f"\n🔍 Sample columns (first 20):")
            for col in combined_df.columns[:20]:
                logger.info(f"   - {col}")
            if len(combined_df.columns) > 20:
                logger.info(f"   ... and {len(combined_df.columns) - 20} more columns")
            
            # Show data sources
            if 'source' in combined_df.columns:
                logger.info(f"\n📊 Data sources:")
                for source, count in combined_df['source'].value_counts().items():
                    logger.info(f"   - {source}: {count} records")
            
            if 'province' in combined_df.columns:
                logger.info(f"\n🗺️  Provinces:")
                for prov, count in combined_df['province'].value_counts().items():
                    logger.info(f"   - {prov}: {count} records")
            
            logger.info(f"\n📋 Preview (first 5 rows):")
            print("\n", combined_df.head())
            
            logger.info(f"\n💡 Next Steps:")
            logger.info(f"   1. Run transformation script to convert to standard format")
            logger.info(f"   2. python src/data/transform_real_data.py")
            logger.info("="*80)
            
            return combined_df
        else:
            logger.warning("\n⚠️  No data collected")
            logger.info("\n💡 Alternative: Use sample data or try manual downloads from:")
            logger.info("   - https://open.canada.ca/")
            logger.info("   - https://catalogue.data.gov.bc.ca/")
            logger.info("   - https://data.ontario.ca/")
            return None
            
    except Exception as e:
        logger.error(f"\n❌ Error during collection: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()
