# PhD Altimetry - Southern Ocean Dynamic Ocean Topography Processing

This repository contains the processing scripts used to generate the **Gridded Altimetric Dynamic Ocean Topography for the Southern Ocean** dataset, supporting the University of Southampton doctoral thesis "Dynamics of the subpolar southern ocean response to climate change" by Oana Claudia Dragomir.

## Dataset Overview

The scripts in this repository process satellite altimetry data to create a Southern Ocean product providing dynamic ocean topography south of 50°S, including over perennialy ice-covered regions. The processing pipeline combines data from two European Space Agency satellite missions:

- **Envisat**
- **CryoSat-2**

### Key Features

- **Temporal Coverage**: July 2002 to October 2018
- **Spatial Coverage**: South of 50°S (Southern Ocean)
- **Spatial Resolution**: 0.5° latitude × 1° longitude grid
- **Temporal Resolution**: Monthly averages
- **Filtering**: Gaussian filter with 300 km radius applied
- **Ice Coverage**: Includes processing over ice-covered regions

## Data Processing Pipeline

The scripts process along-track Sea Surface Height data (referenced to WGS84 ellipsoid) provided by the Centre of Polar Observation and Modelling (CPOM), University College London.

The processing generates Dynamic Ocean Topography (DOT) estimates referenced to three different geoid products:

1. **GOCO05c**
2. **EGM2008** 
3. **EIGEN6S4v2** 

## Repository Structure

```
PhD_altimetry/
├── analysis_functions (helper functions and wrappers)
├── A_altimetry (main processing)
├── B_sea_ice (processing of sea ice data for plotting)
├── C_climate_indices (processing of climate indices for validation)
└── README.md
```

*Detailed script descriptions and workflow documentation will be provided in the upcoming handbook.*

## Dataset Access

The final processed dataset is available through the University of Southampton data repository:

**DOI**: [10.5258/SOTON/D3006](http://dx.doi.org/10.5258/SOTON/D3006)

**Dataset URL**: [https://eprints.soton.ac.uk/492039/](https://eprints.soton.ac.uk/492039/)

### Dataset Files

The dataset includes three NetCDF files, one for each geoid reference:
- `dot_all_30bmedian_goco05c_sig3.nc`
- `dot_all_30bmedian_egm08_sig3.nc`  
- `dot_all_30bmedian_eigen6s4v2_sig3.nc`

## Documentation

📖 **Handbook Coming Soon**: A comprehensive handbook with detailed descriptions of all scripts, processing steps, and workflow will be linked here soon.

## Keywords

- Southern Ocean Sea Level
- Satellite Altimetry
- Sea Ice Cover
- Dynamic Ocean Topography
- Climate Change
- Oceanography

## Citation

If you use this code or the associated dataset, please cite:

```
Dragomir, Oana Claudia (2024): Dataset supporting the University of Southampton doctoral thesis "Dynamics of the subpolar southern ocean response to climate change". University of Southampton. doi:10.5258/SOTON/D3006
```

## Author

**Oana Claudia Dragomir**  
University of Southampton  
PhD Thesis: "Dynamics of the subpolar southern ocean response to climate change"

## License

https://creativecommons.org/licenses/by/4.0/

---

Some utility functions used in this project are adapted from open-source repositories. Special thanks to Clement Vic and Paul Holland for the Antarctic coastline product. Core algorithms and analyses are original contributions developed during my PhD research at the University of Southampton, prior to the widespread adoption of AI tools.
