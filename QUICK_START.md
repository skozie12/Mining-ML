# 🚀 Quick Start: Web Scraping Canadian Mining Data

## TL;DR - Get Started in 3 Steps

### Step 1: Run the Interactive Tool
```bash
python collect_data.py
```

### Step 2: Choose Option 2 (Manual Sources)
This will show you the best websites to download data from.

### Step 3: Or Generate Sample Data
Choose Option 3 to create 1000 test records instantly.

---

## 📥 Top 3 Data Sources (Direct Links)

### 1. 🥇 BC Open Data - Major Mines
**URL**: https://catalogue.data.gov.bc.ca/dataset/major-mines  
**Why**: Clean CSV, well-maintained, comprehensive  
**What**: Active mines, locations, operators, status

### 2. 🥈 Open Canada - Major Projects
**URL**: https://open.canada.ca/data/en/dataset/major-projects  
**Why**: Federal data, covers all provinces  
**What**: Major mining projects nationwide

### 3. 🥉 Impact Assessment Registry
**URL**: https://iaac-aeic.gc.ca/050/evaluations  
**Why**: Detailed project info, includes decisions  
**What**: Environmental assessments, approval status

---

## 🎯 Your Data Should Look Like This

```csv
permit_id,application_date,province,mining_type,mineral_type,company_size,project_area,estimated_duration,distance_to_water,distance_to_protected_area,expected_employment,approved
PM-00001,2024-01-15,Ontario,Open-pit,Gold,Large,250.5,10,5.2,15.0,150,1
PM-00002,2024-02-20,British Columbia,Underground,Copper,Medium,100.0,8,1.5,8.0,75,0
```

---

## 🛠️ Files You Have Now

| File | Purpose |
|------|---------|
| `collect_data.py` | Interactive tool - **START HERE** |
| `src/data/web_scraper.py` | Automated scraping code |
| `DATA_SOURCES.md` | Complete sources guide (read this!) |
| `SCRAPING_GUIDE.md` | Full documentation |

---

## ⚡ Quick Commands

```bash
# Interactive menu
python collect_data.py

# Generate 1000 sample records
python collect_data.py
# Then choose option 3

# Show data sources
python collect_data.py
# Then choose option 2

# Run scraper directly (advanced)
python src/data/web_scraper.py
```

---

## 🎓 Learning Path

1. **Beginner**: Use Option 3 to generate sample data → Start building model
2. **Intermediate**: Use Option 2 → Download from BC Open Data → Load and explore
3. **Advanced**: Customize `src/data/web_scraper.py` for automated scraping

---

## 🆘 Common Issues

**"No pip command"** → Already fixed! Virtual env is set up  
**"No data collected"** → Use manual download (Option 2)  
**"Import errors"** → Already fixed! Packages installed  
**"Where to start?"** → Run `python collect_data.py`

---

## 📞 Need Help?

1. Read `SCRAPING_GUIDE.md` for detailed instructions
2. Read `DATA_SOURCES.md` for all available sources
3. Check error messages in terminal
4. Try sample data first (Option 3)

---

## ✅ Ready to Go!

```bash
python collect_data.py
```

**Choose Option 3** to generate sample data and start building your model right away! 🚀
