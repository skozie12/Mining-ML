# Canadian Mining Data Sources Guide

This guide provides information on where to find and how to scrape Canadian mining permit data.

## 🎯 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Scraper
```bash
python src/data/web_scraper.py
```

## 📊 Public Data Sources

### Federal Sources

#### 1. **Natural Resources Canada (NRCan)**
- **URL**: https://www.nrcan.gc.ca/mining-materials
- **Data**: Major mining projects, minerals production, exploration spending
- **Format**: CSV, Excel, PDF reports
- **How to Access**: 
  - Visit the publications section
  - Download annual reports and datasets
  - Check the "Mining Information" portal

#### 2. **Open Government Portal**
- **URL**: https://open.canada.ca/
- **Search Terms**: "mining", "permits", "mineral exploration"
- **Data Types**: Geospatial data, statistics, project lists
- **Format**: CSV, JSON, Shapefiles, APIs

#### 3. **Impact Assessment Agency of Canada**
- **URL**: https://www.canada.ca/en/impact-assessment-agency.html
- **Registry**: https://iaac-aeic.gc.ca/050/evaluations
- **Data**: Environmental assessments for major projects
- **Includes**: Mining project status, timelines, decisions

### Provincial Sources

#### 4. **British Columbia - MineSpace**
- **URL**: https://mines.nrs.gov.bc.ca/
- **Portal**: https://catalogue.data.gov.bc.ca/
- **Data**: Mining permits, claims, major mines
- **Format**: CSV, Shapefiles, API
- **Note**: BC has excellent open data

#### 5. **Ontario Mining**
- **URL**: http://www.gisapplication.lrc.gov.on.ca/CLAIMaps/
- **Data**: Mining claims, exploration, assessment reports
- **Format**: Interactive maps, downloadable data
- **API**: Ontario GeoHub may have APIs

#### 6. **Quebec - GESTIM/SIGÉOM**
- **URL**: https://gestim.mines.gouv.qc.ca/
- **SIGÉOM**: https://sigeom.mines.gouv.qc.ca/
- **Data**: Mining titles, geological data, exploration
- **Format**: Maps, reports (may require account)
- **Language**: Primarily French

#### 7. **Saskatchewan - Regional Geological and Mineral Deposit Data**
- **URL**: https://www.saskatchewan.ca/business/agriculture-natural-resources-and-industry/mineral-exploration-and-mining
- **Data**: Mining disposition information
- **Format**: PDF reports, mapping applications

#### 8. **Alberta Energy and Minerals**
- **URL**: https://www.alberta.ca/minerals-and-mining
- **Data**: Coal mining, mineral resources
- **Format**: Maps and reports

#### 9. **Yukon Mining**
- **URL**: https://yukon.ca/en/doing-business/industry-operating-yukon/mineral-exploration-and-mining
- **Data**: Mining claims, placer mining, quartz mining
- **Format**: Downloadable datasets

#### 10. **Northwest Territories & Nunavut**
- **URL**: https://www.miningnorth.com/
- **Federal responsibility, territorial involvement
- **Data**: Major project updates, exploration data

## 🤖 Automated Scraping vs Manual Download

### When to Use Automated Scraping:
- ✅ Data updates frequently
- ✅ Website has consistent structure
- ✅ Need real-time updates
- ✅ Website terms of service allow it

### When to Use Manual Download:
- ✅ Large datasets available as bulk downloads
- ✅ APIs are provided
- ✅ Data doesn't change frequently
- ✅ Website structure changes often

## 📋 Recommended Approach

### Step 1: Start with Open Data Downloads
Most provinces offer bulk data downloads. This is the **recommended approach**:

```python
# Use the provided web_scraper.py
python src/data/web_scraper.py
```

### Step 2: Check the Manual Sources
The scraper will provide links to manual download sources. Visit these and download datasets:

1. **BC Open Data**: https://catalogue.data.gov.bc.ca/dataset/major-mines
2. **Open Canada**: Search for "mining projects"
3. **Provincial mining ministries**: Look for "data downloads" or "publications"

### Step 3: Process Downloaded Data
Once you have CSV/Excel files, place them in `data/raw/` and use:

```python
from src.data.data_collection import load_permit_data
from pathlib import Path

# Load your downloaded data
data_path = Path("data/raw/your_downloaded_file.csv")
df = load_permit_data(data_path)
```

## 🔧 Customizing the Scraper

### Add New Data Sources

Edit `src/data/web_scraper.py` and add a new method:

```python
def scrape_new_source(self) -> pd.DataFrame:
    """Scrape from a new source."""
    url = "https://example.com/mining-data"
    response = self.session.get(url)
    # Parse and return DataFrame
    return pd.DataFrame()
```

### Adjust Request Delays

```python
scraper = CanadianMiningDataScraper(delay=2.0)  # 2 seconds between requests
```

## ⚠️ Important Considerations

### Legal & Ethical
1. **Check Terms of Service**: Always review website ToS before scraping
2. **Respect robots.txt**: Check `https://website.com/robots.txt`
3. **Rate Limiting**: Use delays between requests (1-2 seconds minimum)
4. **Attribution**: Cite data sources in your analysis

### Technical
1. **Authentication**: Some portals require login/API keys
2. **Dynamic Content**: Some sites use JavaScript (may need Selenium)
3. **Data Quality**: Always validate scraped data
4. **Updates**: Website structures change - scraper may need updates

### Best Practices
1. **Cache Data**: Save raw scraped data before processing
2. **Version Control**: Track data collection dates
3. **Metadata**: Record source, date, and method for each dataset
4. **Validation**: Cross-reference with official statistics

## 📞 API Access

Some provinces offer official APIs:

### BC Geographic Services
```python
# Example: BC Data Catalogue API
import requests

api_url = "https://catalogue.data.gov.bc.ca/api/3/action/package_search"
params = {"q": "mining", "rows": 10}
response = requests.get(api_url, params=params)
data = response.json()
```

### Federal Open Data API
```python
# Example: Open Canada API
api_url = "https://open.canada.ca/data/api/action/package_search"
params = {"q": "mining permits", "rows": 10}
response = requests.get(api_url, params=params)
```

## 📈 Expected Data Schema

After scraping/downloading, aim to transform data to this format:

```csv
permit_id,application_date,province,mining_type,mineral_type,company_size,
project_area,estimated_duration,distance_to_water,distance_to_protected_area,
expected_employment,approved
```

Use the preprocessing scripts to standardize different source formats.

## 🔄 Data Pipeline

```
1. Scrape/Download → data/raw/
2. Preprocess → data/processed/
3. Feature Engineering → features/
4. Model Training → models/
5. Predictions → results/
```

## 📚 Additional Resources

- **Mining Association of Canada**: https://mining.ca/
- **Fraser Institute Mining Survey**: Annual reports on mining investment
- **PDAC**: Prospectors & Developers Association of Canada
- **Provincial Mining Associations**: Each province has industry associations

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Import errors for `requests` or `beautifulsoup4`
```bash
pip install requests beautifulsoup4 lxml
```

**Issue**: SSL Certificate errors
```python
response = self.session.get(url, verify=False)  # Not recommended for production
```

**Issue**: Timeout errors
```python
response = self.session.get(url, timeout=30)  # Increase timeout
```

**Issue**: Website blocks requests
- Add appropriate User-Agent header (already included in scraper)
- Increase delays between requests
- Consider using their official API or downloading bulk data instead

## 💡 Next Steps

1. Run the scraper to see available sources
2. Manually download recommended datasets
3. Place files in `data/raw/`
4. Run preprocessing pipeline
5. Begin model training

For questions or issues, check the GitHub repository or create an issue.
