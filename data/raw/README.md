# Raw Data Directory

Place your raw mining permit data files here.

## Expected Data Files:
- `permits.csv`: Main permit applications data
- `sample_permits.csv`: Auto-generated sample data for testing

## Data Format:
The data should include columns such as:
- permit_id: Unique identifier
- application_date: Date of application
- province: Canadian province/territory
- mining_type: Type of mining operation
- mineral_type: Type of mineral/resource
- company_size: Size of the company
- project_area: Area of the project (hectares)
- estimated_duration: Expected project duration (years)
- distance_to_water: Distance to nearest water body (km)
- distance_to_protected_area: Distance to protected area (km)
- expected_employment: Expected number of jobs
- approved: Target variable (1=approved, 0=rejected)

Run `python src/data/data_collection.py` to generate sample data.
