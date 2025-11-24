# Lateral-Cephalogram-App
App for accurate lateral cephalometric analysis

Download the Setup Wizard for the App: https://www.dropbox.com/scl/fi/fv5nwns32get5lmemrcdk/LateralCephalogramSetup.exe?rlkey=v4ylb9p9a0nai5jzktk4owfpx&st=uc4ah4zn&dl=1

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Landmarks Supported](#landmarks-supported)
- [Measurements](#measurements)
- [Installation](#installation)
- [Usage Instructions](#usage-instructions)
- [Troubleshooting](#troubleshooting)
- [Research Applications](#research-applications)
- [Contributing](#contributing)
- [Technical Notes](#technical-notes)
- [License](#license)
- [Contact](#contact)

## Overview
This application is a specialized tool developed for PhD dental research to facilitate fast and accurate lateral cephalometric analysis. The tool provides a user-friendly interface for marking anatomical landmarks on X-ray images and automatically calculates relevant measurements, angles, and ratios used in orthodontic and maxillofacial research.

This work was asked of me by a friend who is undertaking the research, and so I created this custom tool for them. It is not my intention to support this tool for others, however I am willing to take questions from others if they find the tool useful. For example, it is likely that the measurements in the tool does not align with the measurements another student/dentist regularly uses. I am happy to make adjustments for individuals that request any such custom changes within reason, as well as grant anyone viewing/using this code to make changes to the code to facilitate their own needs. Contact details are at the bottom of this file.

## Features
- **Interactive Image Analysis**: Load lateral cephalometric radiographs and mark key anatomical landmarks
- **Comprehensive Measurements**: Automatically calculates:
  - Angular measurements (SNA, SNB, ArGoMe, etc.)
  - Linear measurements with proper scaling
  - Perpendicular distances between points and reference lines
  - Ratios and derived calculations relevant to orthodontic analysis
- **Visual Verification**: Displays all reference lines, angles, and perpendicular measurements visually overlaid on the image
- **Data Export**: Save results as CSV files for further analysis and research
- **Results Documentation**: Save annotated images with all measurements for inclusion in research documentation

## Landmarks Supported
The tool supports marking of 35 standard cephalometric landmarks, including:
- S (Sella), N (Nasion), A Point, B Point
- Ar (Articulare), Go (Gonion), Me (Menton)
- Dental landmarks: U1_tip, U1_apex, L1_tip, L1_apex, etc.
- Soft tissue landmarks: ul (upper lip), ll (lower lip), n (soft tissue nasion), pg (soft tissue pogonion)

## Measurements
The following measurements are automatically calculated:
1. **Angular Measurements**:
   - SNA, SNB (skeletal relationships)
   - ArGoMe, ArGoN, NGoMe (mandibular angles)
   - U1/NL, L1/ML (dental inclinations)
   - Interincisal angle (U1/L1)
   - And many more

2. **Linear Measurements**:
   - Maxillary and mandibular lengths
   - Facial heights
   - Dental heights
   - Wits appraisal
   - Soft tissue measurements

3. **Ratios and Calculated Values**:
   - Maxillary and mandibular normative values
   - Proportional relationships between facial components

## Installation

### Prerequisites
- Python 3.6 or higher
- Operating System: Windows, macOS, or Linux
- Sufficient memory for image processing (recommended: 4GB+ RAM)

### Supported Image Formats
- PNG
- JPEG/JPG
- TIFF
- BMP
- Other formats supported by matplotlib

### Install Dependencies
```bash
pip install matplotlib numpy tabulate
```

**Note**: `tkinter` is included with most Python installations. If you encounter issues, you may need to install it separately depending on your system.

### Run the Application
```bash
python lateralceph_app.py
```

## Usage Instructions

### Basic Workflow
1. Launch the application by running `python lateralceph_app.py`
2. Set analysis parameters:
   - Configure directories for input images and output data (Options menu)
   - Set the scale for accurate measurements (Options > Set Scale)
3. Load an image for analysis (File > Load Image)
4. Mark landmarks sequentially as prompted (press spacebar to select a point, Enter to confirm)
5. The application will automatically calculate and display all measurements
6. Save the annotated image and measurement data (File > Save Plot)

### Directory Setup
For efficient workflow, configure these directories:
- Load Image Directory: Location of source radiograph images
- CSV Directory: Location where measurement data will be saved
- Image Directory: Location where annotated images will be saved

### Landmark Marking Sequence
Follow the on-screen prompts to mark landmarks in the specified order. The application requires all 35 landmarks to be placed correctly for accurate analysis.

## Troubleshooting

### Common Issues

**Application won't start:**
- Ensure all dependencies are installed: `pip install matplotlib numpy tabulate`
- Check that you're using Python 3.6 or higher: `python --version`

**Image won't load:**
- Verify the image file format is supported (PNG, JPEG, TIFF, BMP)
- Check that the image file is not corrupted
- Ensure you have read permissions for the image file

**Measurements seem incorrect:**
- Verify that the scale has been set correctly (Options > Set Scale)
- Ensure all 35 landmarks have been placed accurately
- Check that the image orientation is correct (lateral view)

**Cannot save results:**
- Verify you have write permissions to the output directories
- Check that the specified directories exist
- Ensure sufficient disk space is available

## Research Applications
This tool was developed specifically for PhD dental research to standardize cephalometric analysis and maximize workflow efficiency for both lat ceph analysis, and subsequent data analysis.

## Contributing
This tool is primarily developed for specific research purposes, but contributions to improve functionality are welcome. Please contact the repository owner for more information.

## Technical Notes

### Code Structure
You may notice a large number of undeclared variables in the code, corresponding to the names of the 35 landmarks. This is intentional. These variables are dynamically created by the code at runtime, and while it can be removed by simply declaring the variables explicitly in the global variables section, this has no effect on the performance of the code, and it makes for a satisfying demonstration of Python's flexibility in this respect.

Why this method? Mostly for readability, and the desire to remove clunky abstraction/the need to have to reference the landmarks via a clunky reference to the dictionary itself each time, which I disliked. This approach prioritizes code readability while leveraging Python's dynamic variable creation capabilities.

## License
This software is provided for research and educational purposes. Please contact the author for commercial use or redistribution permissions.

## Contact
pennythomas@hotmail.co.uk
