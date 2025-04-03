# Lateral-Cephalogram-App
App for accurate lateral cephalometric analysis

# Lateral Cephalometric Analysis Tool

## Overview
This application is a specialized tool developed for PhD dental research to facilitate fast and accurate lateral cephalometric analysis. The tool provides a user-friendly interface for marking anatomical landmarks on X-ray images and automatically calculates relevant measurements, angles, and ratios used in orthodontic and maxillofacial research.

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
- Required Python packages:
  - matplotlib
  - numpy
  - tkinter
  - tabulate
  - math
  - csv

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/lateral-ceph-analysis.git
   cd lateral-ceph-analysis
   ```

2. Install required dependencies:
   ```
   pip install matplotlib numpy tabulate
   ```
   
3. Run the application:
   ```
   python testing.py
   ```

## Usage Instructions

### Basic Workflow
1. Launch the application
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

## Research Applications
This tool was developed specifically for PhD dental research to standardize cephalometric analysis and facilitate:
- Longitudinal studies of growth and development
- Treatment outcome assessment
- Comparative analysis between different patient groups
- Correlation of cephalometric measurements with clinical outcomes

## Contributing
This tool is primarily developed for specific research purposes, but contributions to improve functionality are welcome. Please contact the repository owner for more information.

## License
[Specify your license information here]

## Citation
If you use this tool in your research, please cite:
[Provide citation information here]

## Contact
[Your contact information]
