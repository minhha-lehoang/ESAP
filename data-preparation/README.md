# Data Preparation
Follow the steps below to extract the MIMIC-III data and align the ScAN annotations with the extracted data.

1. Download the MIMIC-III dataset from the [PhysioNet website](https://mimic.physionet.org/). You will need to request access to the dataset and complete the required training.
   
2. Clone the ScAN repository
```bash
git clone https://github.com/bsinghpratap/ScAN.git
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. Run the `get_HADM_files` script in the ScAN repository.

5. Run the following script to extract the MIMIC-III data and align the annotations with the extracted data. Please replace the corresponding paths in the script with the paths to the MIMIC-III dataset and the ScAN repository.
```bash
python annot-text-extraction.py
```

6. Run the following script to segment notes text into sentences and save the extracted data.
```bash
python notes-segmentation.py
```

