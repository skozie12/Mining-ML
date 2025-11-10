"""
NRCan Canadian Mining and Mineral Processing Atlas Scraper
Focuses on mining operations and mineral processing facilities across         logger.info("\n🔍 Searching for NRCan Mines Database (All Provinces)...")
        
        search_terms = [
            "national inventory abandoned mines",
            "principal mineral areas canada",
            "major mines canada",
            "canadian mines nrcan",
            "active mines canada",
            "mineral deposits canada",
            "mining operations canada",
            "mines ontario nrcan",
            "mines british columbia nrcan",
            "mines quebec nrcan",
            "mines alberta nrcan"
        ]Atlas URL: https://atlas.gc.ca/mins/en/index.html

This scraper accesses NRCan's actual data services including:
- Principal Mineral Areas
- Active and Historical Mines
- Mineral Processing Plants
- Mining Operations by Province
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
from urllib.parse import urljoin

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NRCanAtlasMiningCollector:
    """
    Collector for NRCan Mining Atlas data.
    Accesses GeoGratis, Open Maps, and NRCan data services.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # NRCan data endpoints
        self.nrcan_endpoints = {
            'geogratis': 'https://ftp.maps.canada.ca/pub/',
            'open_canada': 'https://open.canada.ca/data/api/3/action/',
            'atlas': 'https://atlas.gc.ca/'
        }
    
    def get_nrcan_principal_mineral_areas(self):
        """
        Get NRCan Principal Mineral Areas data.
        This is a key dataset showing major mining operations across Canada.
        """
        logger.info("🔍 Accessing NRCan Principal Mineral Areas...")
        
        try:
            # Search for Principal Mineral Areas dataset
            search_url = f"{self.nrcan_endpoints['open_canada']}package_search"
            
            params = {
                "q": "principal mineral areas nrcan",
                "rows": 5,
                "fq": "organization:nrcan-rncan"
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if data['success'] and data['result']['results']:
                for dataset in data['result']['results']:
                    logger.info(f"   Found: {dataset['title']}")
                    
                    # Look for downloadable resources
                    for resource in dataset.get('resources', []):
                        res_format = resource.get('format', '').upper()
                        res_url = resource.get('url', '')
                        res_name = resource.get('name', 'Unknown')
                        
                        logger.info(f"      Resource: {res_format} - {res_name[:50]}")
                        
                        # Try to download any structured data
                        if res_format in ['CSV', 'JSON', 'GEOJSON', 'KML', 'SHP']:
                            try:
                                logger.info(f"      ⬇️  Attempting download...")
                                
                                if res_format == 'CSV':
                                    df = pd.read_csv(res_url, nrows=200)
                                elif res_format in ['JSON', 'GEOJSON']:
                                    response = self.session.get(res_url, timeout=30)
                                    json_data = response.json()
                                    
                                    # Handle GeoJSON features
                                    if 'features' in json_data:
                                        records = []
                                        for feature in json_data['features'][:200]:
                                            props = feature.get('properties', {})
                                            geom = feature.get('geometry', {})
                                            
                                            # Add coordinates
                                            if geom.get('coordinates'):
                                                props['longitude'] = geom['coordinates'][0]
                                                props['latitude'] = geom['coordinates'][1]
                                            
                                            records.append(props)
                                        df = pd.DataFrame(records)
                                    else:
                                        df = pd.DataFrame(json_data)
                                
                                if not df.empty:
                                    logger.info(f"      ✅ Loaded {len(df)} records from NRCan Principal Mineral Areas")
                                    df['source'] = 'NRCan Principal Mineral Areas'
                                    return df
                                    
                            except Exception as e:
                                logger.warning(f"      ⚠️  Could not load: {str(e)[:80]}")
                                continue
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error accessing NRCan data: {e}")
            return pd.DataFrame()
    
    def get_nrcan_mines_database(self):
        """
        Search for NRCan's comprehensive mines database.
        """
        logger.info("\n� Searching for NRCan Mines Database...")
        
        search_terms = [
            "mines database nrcan",
            "canadian mines nrcan",
            "active mines canada nrcan",
            "mineral deposits canada nrcan",
            "mining operations nrcan"
        ]
        
        all_data = []
        
        for term in search_terms:
            try:
                search_url = f"{self.nrcan_endpoints['open_canada']}package_search"
                params = {
                    "q": term,
                    "rows": 5,
                    "fq": "organization:nrcan-rncan AND (res_format:CSV OR res_format:JSON)"
                }
                
                response = self.session.get(search_url, params=params, timeout=15)
                if response.status_code != 200:
                    continue
                    
                data = response.json()
                
                if data['success'] and data['result']['results']:
                    for dataset in data['result']['results']:
                        title = dataset.get('title', '')
                        
                        # Filter for mining-specific datasets
                        if any(kw in title.lower() for kw in ['mine', 'mineral', 'mining', 'deposit']):
                            logger.info(f"   ✅ Found: {title[:60]}")
                            
                            for resource in dataset.get('resources', []):
                                if resource.get('format', '').upper() in ['CSV', 'JSON']:
                                    try:
                                        url = resource['url']
                                        logger.info(f"      ⬇️  Downloading {resource['format']}...")
                                        
                                        if resource['format'].upper() == 'CSV':
                                            # Get ALL records instead of limiting to 150
                                            df = pd.read_csv(url, low_memory=False, on_bad_lines='skip')
                                        else:
                                            df = pd.read_json(url)
                                        
                                        if not df.empty:
                                            df['source'] = title
                                            df['source_org'] = 'NRCan'
                                            all_data.append(df)
                                            logger.info(f"      ✅ Collected {len(df)} records")
                                            
                                            # Check jurisdiction distribution
                                            if 'Jurisdiction' in df.columns:
                                                logger.info(f"      📊 Jurisdictions: {df['Jurisdiction'].value_counts().to_dict()}")
                                            break
                                    except Exception as e:
                                        logger.warning(f"      ⚠️  Error: {str(e)[:60]}")
                                        continue
                
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"   Error with search '{term}': {str(e)[:60]}")
                continue
        
        return all_data
    
    def get_statcan_mining_data(self):
        """
        Get Statistics Canada mining-related data.
        """
        logger.info("\n🔍 Searching Statistics Canada mining data...")
        
        try:
            search_url = "https://open.canada.ca/data/api/3/action/package_search"
            params = {
                "q": "mining mineral production",
                "rows": 5,
                "fq": "organization:statcan"
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                
                if data['success'] and data['result']['results']:
                    for dataset in data['result']['results']:
                        title = dataset.get('title', '')
                        
                        if any(kw in title.lower() for kw in ['mining', 'mineral', 'mine']):
                            logger.info(f"   Found: {title[:60]}")
                            
                            for resource in dataset.get('resources', []):
                                if resource.get('format', '').upper() == 'CSV':
                                    try:
                                        df = pd.read_csv(resource['url'], nrows=100)
                                        if not df.empty:
                                            df['source'] = 'Statistics Canada - ' + title[:40]
                                            logger.info(f"   ✅ Collected {len(df)} StatCan records")
                                            return df
                                    except:
                                        continue
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"StatCan error: {e}")
            return pd.DataFrame()
    
    def get_provincial_mining_data(self):
        """
        Collect mining data from provincial sources - focused on mines only.
        """
        logger.info("\n🗺️  Collecting provincial mining-specific data...")
        
        all_data = []
        
        # British Columbia - Major Mines
        logger.info("\n📍 British Columbia - Major Mines")
        try:
            bc_api = "https://catalogue.data.gov.bc.ca/api/3/action/package_search"
            params = {"q": "major mines operating", "rows": 5}
            
            response = self.session.get(bc_api, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    for dataset in data['result']['results']:
                        title = dataset.get('title', '').lower()
                        
                        # Only get mine-specific datasets
                        if 'mine' in title and 'forest' not in title:
                            logger.info(f"   Found: {dataset.get('title', 'Unknown')}")
                            
                            for resource in dataset.get('resources', []):
                                if resource['format'].upper() == 'CSV':
                                    try:
                                        df = pd.read_csv(resource['url'], nrows=150)
                                        df['province'] = 'British Columbia'
                                        df['source'] = 'BC Major Mines'
                                        all_data.append(df)
                                        logger.info(f"   ✅ Collected {len(df)} BC mine records")
                                        break
                                    except:
                                        continue
        except Exception as e:
            logger.warning(f"   ⚠️  BC error: {e}")
        
        time.sleep(1)
        
        # Ontario - Mining Data
        logger.info("\n📍 Ontario - Mining Operations")
        try:
            ont_api = "https://data.ontario.ca/api/3/action/package_search"
            params = {"q": "mining mine mineral", "rows": 5}
            
            response = self.session.get(ont_api, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data['success']:
                    for dataset in data['result']['results']:
                        title = dataset.get('title', '').lower()
                        
                        # Only mining-specific
                        if ('mine' in title or 'mineral' in title) and 'forest' not in title:
                            logger.info(f"   Found: {dataset.get('title', 'Unknown')}")
                            
                            for resource in dataset.get('resources', []):
                                if resource['format'].upper() in ['CSV', 'JSON']:
                                    try:
                                        if resource['format'].upper() == 'CSV':
                                            df = pd.read_csv(resource['url'], nrows=150)
                                        else:
                                            df = pd.read_json(resource['url'])
                                            df = df.head(150)
                                        
                                        df['province'] = 'Ontario'
                                        df['source'] = 'Ontario Mining Data'
                                        all_data.append(df)
                                        logger.info(f"   ✅ Collected {len(df)} Ontario mine records")
                                        break
                                    except:
                                        continue
        except Exception as e:
            logger.warning(f"   ⚠️  Ontario error: {e}")
        
        time.sleep(1)
        
        # Quebec - GESTIM Mining Data
        logger.info("\n📍 Quebec - Mining Sites")
        try:
            # Quebec's SIGÉOM open data
            qc_urls = [
                "https://gq.mines.gouv.qc.ca/documents/SIGEOM/TOUTQC/FRA/CSV/SIGEOM_QC_Potentiel_mineral.csv",
                "https://gq.mines.gouv.qc.ca/documents/SIGEOM/TOUTQC/FRA/CSV/SIGEOM_QC_Gisements.csv"
            ]
            
            for url in qc_urls:
                try:
                    logger.info(f"   Downloading from SIGÉOM...")
                    df = pd.read_csv(url, nrows=150, low_memory=False, encoding='utf-8')
                    df['province'] = 'Quebec'
                    df['source'] = 'Quebec SIGEOM'
                    all_data.append(df)
                    logger.info(f"   ✅ Collected {len(df)} Quebec mining records")
                    break
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not load Quebec data: {str(e)[:50]}")
                    continue
        except Exception as e:
            logger.warning(f"   ⚠️  Quebec error: {e}")
        
        return all_data
    
    def collect_all_mining_data(self, target_records=500):
        """
        Complete collection focusing only on NRCan mining data.
        """
        logger.info("\n" + "="*80)
        logger.info("⛏️  NRCan MINING DATA COLLECTOR")
        logger.info("     Collecting from NRCan and Statistics Canada")
        logger.info("="*80)
        
        all_data = []
        
        # Step 1: Get NRCan Principal Mineral Areas
        logger.info("\n📊 Step 1: NRCan Principal Mineral Areas...")
        mineral_areas = self.get_nrcan_principal_mineral_areas()
        if not mineral_areas.empty:
            all_data.append(mineral_areas)
            logger.info(f"   ✅ Collected {len(mineral_areas)} records")
        
        time.sleep(1)
        
        # Step 2: Search NRCan mines database
        logger.info("\n📊 Step 2: NRCan Mines Database Search...")
        mines_data = self.get_nrcan_mines_database()
        if mines_data:
            all_data.extend(mines_data)
            total = sum(len(df) for df in mines_data)
            logger.info(f"   ✅ Collected {total} records from {len(mines_data)} datasets")
        
        time.sleep(1)
        
        # Step 3: Statistics Canada mining data
        logger.info("\n📊 Step 3: Statistics Canada Mining Data...")
        statcan_data = self.get_statcan_mining_data()
        if not statcan_data.empty:
            all_data.append(statcan_data)
            logger.info(f"   ✅ Collected {len(statcan_data)} StatCan records")
        
        time.sleep(1)
        
        # Step 4: Provincial data if needed
        total_so_far = sum(len(df) for df in all_data)
        logger.info(f"\n   Total collected so far: {total_so_far} records")
        
        if total_so_far < target_records:
            logger.info(f"\n📊 Step 4: Supplementing with Provincial Data...")
            provincial_data = self.get_provincial_mining_data()
            if provincial_data:
                all_data.extend(provincial_data)
        
        return all_data


def main():
    """Main execution function."""
    
    output_path = Path(__file__).parent.parent.parent / "data" / "raw"
    output_path.mkdir(parents=True, exist_ok=True)
    
    collector = NRCanAtlasMiningCollector()
    
    try:
        # Collect mining data
        mining_data = collector.collect_all_mining_data(target_records=500)
        
        if mining_data:
            # Combine all dataframes
            logger.info("\n" + "="*80)
            logger.info("📊 COMBINING MINING DATA")
            logger.info("="*80)
            
            combined_df = pd.concat(mining_data, ignore_index=True, sort=False)
            
            # Save
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_path / f"nrcan_atlas_mining_data_{timestamp}.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"\n✅ SUCCESS!")
            logger.info("="*80)
            logger.info(f"📁 Saved to: {output_file}")
            logger.info(f"📊 Total mining records: {len(combined_df)}")
            logger.info(f"📋 Columns: {len(combined_df.columns)}")
            
            # Show sample columns
            logger.info(f"\n🔍 Sample columns (first 15):")
            for col in combined_df.columns[:15]:
                logger.info(f"   - {col}")
            if len(combined_df.columns) > 15:
                logger.info(f"   ... and {len(combined_df.columns) - 15} more")
            
            # Show sources
            if 'source' in combined_df.columns:
                logger.info(f"\n📊 Data sources:")
                for source, count in combined_df['source'].value_counts().items():
                    logger.info(f"   - {source}: {count} records")
            
            if 'province' in combined_df.columns:
                logger.info(f"\n🗺️  Provinces:")
                for prov, count in combined_df['province'].value_counts().items():
                    logger.info(f"   - {prov}: {count} records")
            
            # Check for mining-specific columns
            mining_cols = [col for col in combined_df.columns 
                          if any(k in str(col).lower() for k in 
                          ['mine', 'mineral', 'commodity', 'ore', 'deposit'])]
            
            if mining_cols:
                logger.info(f"\n⛏️  Mining-specific columns found:")
                for col in mining_cols[:10]:
                    logger.info(f"   - {col}")
            
            logger.info(f"\n💡 Next Steps:")
            logger.info(f"   1. Transform to standard format:")
            logger.info(f"      python src/data/transform_real_data.py")
            logger.info(f"   2. Review the mining data quality")
            logger.info(f"   3. Use for model training")
            logger.info("="*80)
            
            return combined_df
        else:
            logger.warning("\n⚠️  No mining data collected")
            logger.info("\n💡 Alternative: Check NRCan Atlas website manually:")
            logger.info("   https://atlas.gc.ca/mins/en/index.html")
            return None
            
    except Exception as e:
        logger.error(f"\n❌ Error during collection: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    main()
