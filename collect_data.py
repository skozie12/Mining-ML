"""
Example script demonstrating how to scrape and collect Canadian mining data.

This script shows different approaches to collecting data:
1. Automated web scraping
2. Manual download guidance
3. Processing collected data
"""

from pathlib import Path
import logging
from src.data.web_scraper import CanadianMiningDataScraper, scrape_mining_data_manual
from src.data.data_collection import load_permit_data, create_sample_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("🏔️  CANADIAN MINING DATA COLLECTION TOOL")
    print("="*80)
    
    # Set up paths
    raw_data_path = Path("data/raw")
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    print("\n📂 Data will be saved to:", raw_data_path.absolute())
    
    # Menu
    print("\n🎯 Choose an option:")
    print("1. Try automated web scraping (experimental)")
    print("2. Show manual download sources (recommended)")
    print("3. Generate sample data for testing")
    print("4. Load and preview existing data")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "-"*80)
        print("Option 1: Automated Web Scraping")
        print("-"*80)
        print("\n⚠️  Note: Automated scraping may not work for all sources.")
        print("Many Canadian government sites prefer API access or manual downloads.")
        
        confirm = input("\nProceed with automated scraping? (yes/no): ").strip().lower()
        
        if confirm == "yes":
            scraper = CanadianMiningDataScraper(delay=2.0)
            data = scraper.scrape_all_sources(raw_data_path)
            
            if not data.empty:
                print(f"\n✅ Successfully collected {len(data)} records")
                print("\nPreview of collected data:")
                print(data.head())
            else:
                print("\n⚠️  No data collected. Try manual download option instead.")
        else:
            print("Scraping cancelled.")
    
    elif choice == "2":
        print("\n" + "-"*80)
        print("Option 2: Manual Download Sources")
        print("-"*80)
        print("\n📚 Here are the best sources for Canadian mining data:\n")
        
        print("🔹 RECOMMENDED SOURCES:")
        print("\n1. BC Open Data Catalogue")
        print("   URL: https://catalogue.data.gov.bc.ca/")
        print("   Search: 'major mines' or 'mining permits'")
        print("   Format: CSV (easy to download)")
        
        print("\n2. Open Government Canada")
        print("   URL: https://open.canada.ca/")
        print("   Search: 'mining projects' or 'mineral exploration'")
        print("   Format: Various (CSV, JSON, API)")
        
        print("\n3. Natural Resources Canada")
        print("   URL: https://www.nrcan.gc.ca/mining-materials")
        print("   Look for: 'Publications' and 'Data Downloads'")
        print("   Format: Excel, PDF reports")
        
        print("\n4. Impact Assessment Agency")
        print("   URL: https://iaac-aeic.gc.ca/050/evaluations")
        print("   Data: Environmental assessments for major projects")
        print("   Note: Includes detailed project information")
        
        print("\n" + "-"*80)
        print("\n📥 INSTRUCTIONS:")
        print("1. Visit the URLs above")
        print("2. Download CSV/Excel files")
        print("3. Place them in: data/raw/")
        print("4. Run this script again and choose option 4 to preview")
        print("-"*80)
        
        # Run the manual sources function for detailed output
        scrape_mining_data_manual()
        
        print("\n💡 TIP: Read DATA_SOURCES.md for comprehensive guidance!")
    
    elif choice == "3":
        print("\n" + "-"*80)
        print("Option 3: Generate Sample Data")
        print("-"*80)
        
        n_samples = input("\nHow many sample records to generate? (default: 1000): ").strip()
        n_samples = int(n_samples) if n_samples.isdigit() else 1000
        
        print(f"\n🔄 Generating {n_samples} sample mining permit records...")
        sample_data = create_sample_data(raw_data_path, n_samples=n_samples)
        
        print("\n✅ Sample data generated successfully!")
        print(f"📁 Saved to: {raw_data_path}/sample_permits.csv")
        
        print("\n📊 Data Preview:")
        print(sample_data.head(10))
        
        print("\n📈 Data Statistics:")
        print(f"   Total records: {len(sample_data)}")
        print(f"   Columns: {len(sample_data.columns)}")
        print(f"   Provinces: {sample_data['province'].nunique()}")
        print(f"   Mining types: {sample_data['mining_type'].nunique()}")
        print(f"   Average approval time: {sample_data['approval_time_months'].mean():.1f} months")
        
        print("\n💡 You can now use this data for model development and testing!")
    
    elif choice == "4":
        print("\n" + "-"*80)
        print("Option 4: Load Existing Data")
        print("-"*80)
        
        # List available files
        csv_files = list(raw_data_path.glob("*.csv"))
        
        if not csv_files:
            print("\n⚠️  No CSV files found in data/raw/")
            print("Please download data or generate sample data first.")
            return
        
        print("\n📂 Available data files:")
        for i, file in enumerate(csv_files, 1):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"{i}. {file.name} ({size_mb:.2f} MB)")
        
        file_choice = input(f"\nSelect file (1-{len(csv_files)}): ").strip()
        
        if file_choice.isdigit() and 1 <= int(file_choice) <= len(csv_files):
            selected_file = csv_files[int(file_choice) - 1]
            
            print(f"\n📖 Loading: {selected_file.name}...")
            data = load_permit_data(selected_file)
            
            print("\n✅ Data loaded successfully!")
            print(f"\n📊 Dataset Info:")
            print(f"   Records: {len(data)}")
            print(f"   Columns: {len(data.columns)}")
            print(f"   Columns: {', '.join(data.columns.tolist())}")
            
            print("\n📋 First 10 rows:")
            print(data.head(10))
            
            print("\n📈 Summary Statistics:")
            print(data.describe())
        else:
            print("Invalid selection.")
    
    else:
        print("\n❌ Invalid choice. Please run the script again.")
    
    print("\n" + "="*80)
    print("✅ Process complete!")
    print("\n📚 Next steps:")
    print("   - Read DATA_SOURCES.md for detailed guidance")
    print("   - Run preprocessing: python src/data/preprocessing.py")
    print("   - Explore data: Open notebooks/01_data_exploration.ipynb")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")
